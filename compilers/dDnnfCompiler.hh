/*
* d4
* Copyright (C) 2020  Univ. Artois & CNRS
* 
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
* 
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef COMPILERS_DDNNF_COMPILER
#define COMPILERS_DDNNF_COMPILER

#include <iostream>
#include <memory>
#include <vector>
#include <boost/multiprecision/gmp.hpp>

#include "../interfaces/OccurrenceManagerInterface.hh"
#include "../interfaces/PartitionerInterface.hh"
#include "../interfaces/VariableHeuristicInterface.hh"

#include "../manager/dynamicOccurrenceManager.hh"
#include "../manager/BucketManager.hh"
#include "../manager/CacheCNFManager.hh"

#include "../utils/System.hh"
#include "../utils/SolverTypes.hh"
#include "../utils/Dimacs.hh"
#include "../utils/Solver.hh"

#include "../mtl/Sort.hh"
#include "../mtl/Vec.hh"
#include "../mtl/Heap.hh"
#include "../mtl/Alg.hh"

#include "../DAG/UnaryNode.hh"
#include "../DAG/UnaryNodeCertified.hh"
#include "../DAG/BinaryDeterministicOrNode.hh"
#include "../DAG/BinaryDeterministicOrNodeCertified.hh"
#include "../DAG/DecomposableAndNodeCerified.hh"
#include "../DAG/DecomposableAndNode.hh"
#include "../DAG/DAG.hh"

#include "../manager/OptionManager.hh"
#include "../core/ShareStructures.hh"


#define NB_SEP_DNNF_COMPILER 154

using namespace boost::multiprecision;
using namespace std;


struct onTheBranch
{
    vec<Lit> units;
    vec<Var> free;
    vec<int> idxReason;
};

template <class T> class DDnnfCompiler
{
public:
    static constexpr bool COMPRESS = false;
private:
    // statistics
    int nbNodeInCompile;
    int nbCallCompile;
    int nbSplit;
    int callEquiv, callPartitioner;
    double currentTime;

    int freqBackbone;
    double sumAffectedAndNode;
    int minAffectedAndNode;

    int freqLimitDyn;
    unsigned int nbDecisionNode;
    unsigned int nbDomainConstraintNode;
    unsigned int nbAndNode, nbAndMinusNode;
    CacheCNF<std::weak_ptr<DAG<T> > >* cache;

    vec<unsigned> stampVar;
    vec<bool> alreadyAdd;
    unsigned stampIdx;

    bool optBackbone;
    int optCached;
    bool optDecomposableAndNode;
    bool optDomConst;
    bool optReversePolarity;
    bool isCertified;

    VariableHeuristicInterface *vs;
    BucketManager<std::weak_ptr<DAG<T> > > *bm;
    PartitionerInterface *pv;

    EquivManager em;

    std::shared_ptr<DAG<T> > globalTrueNode, globalFalseNode;

    Solver s;
    OccurrenceManagerInterface *occManager;
    vec<vec<Lit> > clauses;

    bool initUnsat;
    TmpEntry<std::weak_ptr<DAG<T> > > NULL_CACHE_ENTRY;


    /**
       Manage the case where it is unsatisfiable.
    */
    std::shared_ptr<DAG<T> > manageUnsat(Lit l, onTheBranch &onB, vec<int> &idxReason)
    {
        // we need to get a reason for why the problem is unsat.
        onB.units.push(l);
        if(!isCertified) return globalFalseNode;
        if(s.idxReasonFinal >= 0) idxReason.push(s.idxReasonFinal);
        return globalFalseNode;
    }// manageUnsat

    /**
       Compile the CNF formula into a D-FPiBDD.

       @param[in] setOfVar, the current set of considered variables
       @param[in] priority, select in priority these variable to the next decision node
       @param[in] dec, the decision literal
       @param[out] onB, information about units, free variables on the branch
       @param[out] fromCache, to know if the DAG returned is from cache
       @param[out] idxReason, the reason for the units (please only add and do not clean this variable, reuse after)

       \return a compiled formula (fpibdd or fbdd w.r.t. the options selected).
    */
    std::shared_ptr<DAG<T> > compile_(vec<Var> &setOfVar, vec<Var> &priorityVar, Lit dec, onTheBranch &onB,
                     bool &fromCache, vec<int> &idxReason)
    {
        fromCache = false;
        showRun(); nbCallCompile++;
        s.rebuildWithConnectedComponent(setOfVar);

        if(!s.solveWithAssumptions()) return manageUnsat(dec, onB, idxReason);
        s.collectUnit(setOfVar, onB.units, dec); // collect unit literals
        occManager->preUpdate(onB.units);

        // compute the connected composant
        vec<Var> reallyPresent;
        vec< vec<Var> > varConnected;
        int nbComponent = occManager->computeConnectedComponent(varConnected, setOfVar, onB.free, reallyPresent);

        if(nbComponent && !optDecomposableAndNode)
        {
            for(int i =  1 ; i<varConnected.size() ; i++)
                for(int j = 0 ; j<varConnected[i].size() ; j++) varConnected[0].push(varConnected[i][j]);
            nbComponent = 1;
        }

        std::vector<bool> comeFromCache;
        std::shared_ptr<DAG<T> > ret = nullptr;
        if(!nbComponent)
        {
            comeFromCache.push_back(false);
            ret = globalTrueNode; // tautologie modulo unit literal
        }
        else
        {
            // we considere each component one by one
            std::vector<std::shared_ptr<DAG<T> > > andDecomposition;

            nbSplit += (nbComponent > 1) ? nbComponent : 0;
            for(int cp = 0 ; cp<nbComponent ; cp++)
            {
                vec<Var> &connected = varConnected[cp];
                bool localCache = optCached;

                occManager->updateCurrentClauseSet(connected);
                TmpEntry<std::weak_ptr<DAG<T> > > cb = (localCache) ? cache->searchInCache(connected, bm) : NULL_CACHE_ENTRY;

                if(localCache && cb.defined && !cb.getValue().expired())
                {
                    comeFromCache.push_back(true);
                    andDecomposition.push_back(cb.getValue().lock());
                }
                else
                {
                    // compute priority list
                    vec<Var> currPriority;
                    comeFromCache.push_back(false);

                    stampIdx++;
                    for(int i = 0 ; i<connected.size() ; i++) stampVar[connected[i]] = stampIdx;
                    bool propagatePriority = 1 || onB.units.size() < (setOfVar.size() / 10);

                    for(int i = 0 ; propagatePriority && i<priorityVar.size() ; i++)
                        if(stampVar[priorityVar[i]] == stampIdx && s.value(priorityVar[i]) == l_Undef && DAG<T>::varProjected[priorityVar[i]]) {
                            currPriority.push(priorityVar[i]);
                        }

                    ret = compileDecisionNode(connected, currPriority);
                    andDecomposition.push_back(ret);
                    //if(localCache && !ret->isUnaryNode()) cache->addInCache(cb, ret);
                    if(localCache) cache->addInCache(cb, ret);
                }
                occManager->popPreviousClauseSet();
            }

            //compress
            if(COMPRESS) {
                std::vector<std::shared_ptr<UnaryNode<T> > > unary;

                for(int i = andDecomposition.size() - 1; i >= 0; i--) {
                    if(andDecomposition[i]->isUnaryNode()) {
                        unary.push_back(std::dynamic_pointer_cast<UnaryNode<T> >(andDecomposition[i]));
                        andDecomposition.erase(andDecomposition.begin() + i);
                        comeFromCache.erase(comeFromCache.begin() + i);
                    }
                }

                std::vector<Var> vFree;
                std::vector<Lit> vUnit;

                while (!unary.empty()) {
                    vFree.clear();
                    vUnit.clear();

                    std::shared_ptr<UnaryNode<T> > u = unary[unary.size() - 1];
                    unary.pop_back();

                    for(Lit l : u->units) {
                        vUnit.push_back(l);
                    }
                    for(Var v : u->free) {
                        vFree.push_back(v);
                    }

                    for(int i = unary.size() - 1; i >= 0; i--) {
                        if(unary[i]->child == u->child) {
                            std::shared_ptr<UnaryNode<T> > tmp = unary[i];
                            unary.erase(unary.begin() + i);

                            for(Lit l : tmp->units) {
                                vUnit.push_back(l);
                            }
                            for(Var v : tmp->free) {
                                vFree.push_back(v);
                            }
                        }
                    }

                    andDecomposition.push_back(std::make_shared<UnaryNode<T> >(u->child, vUnit, vFree));
                    comeFromCache.push_back(false);
                }

                nbComponent = andDecomposition.size();
            }

            assert(nbComponent);
            if(nbComponent <= 1)
            {
                fromCache = comeFromCache[0];
                ret = andDecomposition[0];
            }
            else
            {
                if(isCertified) ret = std::make_shared<DecomposableAndNodeCertified<T> >(andDecomposition, std::move(comeFromCache));
                else ret = std::make_shared<DecomposableAndNode<T> >(andDecomposition);
                nbAndNode++;

                // statistics
                if(minAffectedAndNode > (s.trail).size()) minAffectedAndNode = (s.trail).size();
                sumAffectedAndNode += (s.trail).size();
            }
        }

        assert(ret);
        occManager->postUpdate(onB.units);

        if(isCertified)
        {
            if(s.decisionLevel() != s.assumptions.size()) s.refillAssums();
            for(int i = 0 ; i<setOfVar.size() ; i++)
            {
                Var v = setOfVar[i];
                if(s.value(v) != l_Undef && s.reason(v) != CRef_Undef) idxReason.push(s.ca[s.reason(v)].idxReason());
            }
        } else if(s.decisionLevel() != s.assumptions.size()) s.refillAssums();

        return ret;
    }// compile_


    /**
       Create a decision node in purpose.
    */
    std::shared_ptr<DAG<T> > createObjectDecisionNode(std::shared_ptr<DAG<T> > pos, onTheBranch &bPos, bool fromCachePos,
                                                      std::shared_ptr<DAG<T> > neg, onTheBranch &bNeg, bool fromCacheNeg,
                                     vec<int> &idxReason)
    {
        if(isCertified)
            return std::make_shared<BinaryDeterministicOrNodeCertified<T> >(pos, bPos.units, bPos.free, fromCachePos,
                                                             neg, bNeg.units, bNeg.free, fromCacheNeg, idxReason);
        return std::make_shared<BinaryDeterministicOrNode<T> >(pos, bPos.units, bPos.free, neg, bNeg.units, bNeg.free);
    }// createDecisionNode


    /**
       This function select a variable and compile a decision node.

       @param[in] connected, the set of variable present in the current problem
       \return the compiled formula
    */
    std::shared_ptr<DAG<T> > compileDecisionNode(vec<Var> &connected, vec<Var> &priorityVar)
    {
        if(s.assumptions.size() && s.assumptions.size() < 5){cout << "c top 5: "; showListLit(s.assumptions);}

        bool weCall = false;
        if(pv && !priorityVar.size() && connected.size() > 10 && connected.size() < 5000)
        {
            weCall = true;
            vec<int> cutSet;
            pv->computePartition(connected, cutSet, priorityVar, vs->getScoringFunction());

            // normally priority var is a subset of connect ???
            for(int i = 0 ; i<priorityVar.size() ; i++)
            {
                bool isIn = false;
                for(int j = 0 ; !isIn && j < connected.size() ; j++) isIn = connected[j] == priorityVar[i];
                assert(isIn);
            }

            callPartitioner++;
        }

        Var v = var_Undef;
        if(priorityVar.size()) v = vs->selectVariable(priorityVar); else v = vs->selectVariable(connected);
        // PATCH:
        if(v == var_Undef && priorityVar.size() > 0 && priorityVar.size() < connected.size()) v = vs->selectVariable(connected);
        assert(priorityVar.size() == 0 || v != var_Undef);
        if(v == var_Undef) return createTrueNode(connected);

        Lit l = mkLit(v, optReversePolarity - vs->selectPhase(v));
        nbDecisionNode++;

        onTheBranch bPos, bNeg;
        bool fromCachePos, fromCacheNeg;
        vec<int> idxReason;

        // compile the formula where l is assigned to true
        assert(s.value(l) == l_Undef);
        (s.assumptions).push(l);
        std::shared_ptr<DAG<T> > pos = compile_(connected, priorityVar, l, bPos, fromCachePos, idxReason);
        (s.assumptions).pop();
        (s.cancelUntil)((s.assumptions).size());

        // compile the formula where l is assigned to true
        (s.assumptions).push(~l);
        std::shared_ptr<DAG<T> > neg = compile_(connected, priorityVar, ~l, bNeg, fromCacheNeg, idxReason);

        (s.assumptions).pop();
        (s.cancelUntil)((s.assumptions).size());


        // Compress
        if(COMPRESS) {
            bPos.units.clear();
            bPos.units.push(l);

            bNeg.units.clear();
            bNeg.units.push(~l);

            bool modif = true;

            while(modif) {
                modif = false;

                while (pos->isUnaryNode()) {
                    modif = true;
                    auto u = std::dynamic_pointer_cast<UnaryNode<T> >(pos);
                    bPos.free.capacity(bPos.free.size() + u->free.size());

                    //Var *vf = &DAG<T>::freeVariables[u->branch.idxFreeVar];
                    //for (int i = 0; vf[i] != var_Undef; i++) {
                    //    bPos.free.push(vf[i]);
                    //}
                    for(Var v : u->free) {
                        bPos.free.push(v);
                    }

                    if (pos == u->child) {
                        std::cout << "## pos error " << var(bPos.units[0]) << " (";
                        for (int i = 0; i < bPos.free.size(); i++) {
                            std::cout << " " << i;
                        }
                        std::cout << ")\n";
                        break;
                    }
                    pos = u->child;
                }
                while (neg->isUnaryNode()) {
                    modif = true;
                    auto u = std::dynamic_pointer_cast<UnaryNode<T> >(neg);
                    bNeg.free.capacity(bNeg.free.size() + u->free.size());

                    //Var *vf = &DAG<T>::freeVariables[u->branch.idxFreeVar];
                    //for (int i = 0; vf[i] != var_Undef; i++) {
                    //    bNeg.free.push(vf[i]);
                    //}
                    for (Var v: u->free) {
                        bNeg.free.push(v);
                    }

                    if (neg == u->child) {
                        std::cout << "## neg error " << var(bNeg.units[0]) << " (";
                        for (int i = 0; i < bNeg.free.size(); i++) {
                            std::cout << " " << i;
                        }
                        std::cout << ")\n";
                        break;
                    }
                    neg = u->child;
                }

                //continue;

                if(pos->isAndNode()) {
                    auto a = std::dynamic_pointer_cast<DecomposableAndNode<T> >(pos);

                    for(int i = a->nb_children() - 1; i >= 0; i--) {
                        if((*a)[i]->isUnaryNode()) {
                            modif = true;
                            auto u = std::dynamic_pointer_cast<UnaryNode<T> >((*a)[i]);
                            if(u->child == globalTrueNode) {
                                a->erase(i);
                            }
                            else {
                                (*a)[i] = u->child;
                            }
                            //(*a)[i] = u->child;

                            bPos.free.capacity(bPos.free.size() + u->free.size());
                            for(auto v : u->free) {
                                bPos.free.push(v);
                            }
                        }
                    }

                    if(a->nb_children() == 1) {
                        modif = true;
                        pos = (*a)[0];
                        // comefromcache ???
                    }
                }

                if(neg->isAndNode()) {
                    auto a = std::dynamic_pointer_cast<DecomposableAndNode<T> >(neg);

                    for(int i = a->nb_children() - 1; i >= 0; i--) {
                        if((*a)[i]->isUnaryNode()) {
                            modif = true;
                            auto u = std::dynamic_pointer_cast<UnaryNode<T> >((*a)[i]);
                            if(u->child == globalTrueNode) {
                                a->erase(i);
                            }
                            else {
                                (*a)[i] = u->child;
                            }
                            //(*a)[i] = u->child;

                            bNeg.free.capacity(bNeg.free.size() + u->free.size());
                            for(auto v : u->free) {
                                bNeg.free.push(v);
                            }
                        }
                    }

                    if(a->nb_children() == 1) {
                        modif = true;
                        neg = (*a)[0];
                        // comefromcache ???
                    }
                }
            }

            if (neg == pos && bPos.free == bNeg.free) {
                bPos.units.clear();
                bPos.free.push(var(l));
                if (pos->isUnaryNode()) {
                    auto u = std::dynamic_pointer_cast<UnaryNode<T> >(pos);
                    bPos.free.capacity(bPos.free.size() + u->free.size());

                    //Var *vf = &DAG<T>::freeVariables[u.branch.idxFreeVar];
                    //for (int i = 0; vf[i] != var_Undef; i++) {
                    //    bPos.free.push(vf[i]);
                    //}
                    for(Var v : u->free) {
                        bPos.free.push(v);
                    }

                    pos = u->child;
                }
                auto ret = std::make_shared<UnaryNode<T> >(pos, bPos.units, bPos.free);
                return ret;
            }

            if(neg == globalFalseNode) {
                auto ret = std::make_shared<UnaryNode<T> >(pos, bPos.units, bPos.free);
                return ret;
            }
            if(pos == globalFalseNode) {
                auto ret = std::make_shared<UnaryNode<T> >(neg, bNeg.units, bNeg.free);
                return ret;
            }
        }

        std::shared_ptr<DAG<T> > ret = createObjectDecisionNode(pos, bPos, fromCachePos, neg, bNeg, fromCacheNeg, idxReason);
        return ret;
    }// compileDecisionNode


    inline void showHeader()
    {
        separator();
        printf("c %10s | %10s | %10s | %10s | %10s | %10s | %10s | %10s | %10s | %10s | %11s | %10s | \n",
               "#compile", "time", "#posHit", "#negHit", "#split", "Mem(MB)",
               "#nodes", "#edges", "#equivCall", "#Dec. Node", "#paritioner", "limit dyn");
        separator();
    }


    inline void showInter()
    {
        double now = cpuTime();

        printf("c %10d | %10.2lf | %10d | %10d | %10d | %10.0lf | %10d | %10d | %10d | %10d | %11d | %10d | \n",
               nbCallCompile, now - currentTime, cache->getNbPositiveHit(),
               cache->getNbNegativeHit(), nbSplit, memUsedPeak(), DAG<T>::nbNodes, DAG<T>::nbEdges,
               callEquiv, nbDecisionNode, callPartitioner, 0);
    }

    inline void printFinalStatsCache()
    {
        separator();
        printf("c\n");
        printf("c \033[1m\033[31mStatistics \033[0m\n");
        printf("c \033[33mCompilation Information\033[0m\n");
        printf("c Number of compiled node: %d\n", nbCallCompile);
        printf("c Number of split formula: %d\n", nbSplit);
        printf("c Number of decision node: %u\n", nbDecisionNode);
        printf("c Number of node built on domain constraints: %u\n", nbDomainConstraintNode);
        printf("c Number of decomposable AND nodes: %u\n", nbAndNode);
        printf("c Number of backbone calls: %u\n", callEquiv);
        printf("c Number of partitioner calls: %u\n", callPartitioner);
        printf("c Average number of assigned literal to obtain decomposable AND nodes: %.2lf/%d\n",
               nbAndNode ? sumAffectedAndNode / nbAndNode : s.nVars(), s.nVars());
        printf("c Minimum number of assigned variable where a decomposable AND appeared: %u\n", minAffectedAndNode);
        printf("c \n");
        printf("c \033[33mGraph Information\033[0m\n");
        printf("c Number of nodes: %d\n", DAG<T>::nbNodes);
        printf("c Number of edges: %d\n", DAG<T>::nbEdges);
        printf("c \n");
        cache->printCacheInformation();
        printf("c Final time: %lf\n", cpuTime());
        printf("c \n");
    }// printFinalStat

    inline void showRun()
    {
        if(!(nbCallCompile & (MASK_HEADER))) showHeader();
        if(nbCallCompile && !(nbCallCompile & MASK)) showInter();
    }

    inline void separator(){ printf("c "); for(int i = 0 ; i<NB_SEP_DNNF_COMPILER ; i++) printf("-"); printf("\n");}

    inline std::shared_ptr<DAG<T> > createTrueNode(vec<Var> &setOfVar)
    {
        vec<Lit> unitLit;
        s.collectUnit(setOfVar, unitLit); // collect unit literals
        if(unitLit.size())
        {
            vec<Var> freeVar;
            if(!isCertified) return std::make_shared<UnaryNode<T> >(globalTrueNode, unitLit, freeVar);

            vec<int> idxReason;
            assert(s.decisionLevel() == s.assumptions.size());
            for(int i = 0 ; i<setOfVar.size() ; i++)
            {
                Var v = setOfVar[i];
                if(s.value(v) != l_Undef && s.reason(v) != CRef_Undef) idxReason.push(s.ca[s.reason(v)].idxReason());
            }

            return std::make_shared<UnaryNodeCertified<T> >(globalTrueNode, unitLit, false, idxReason, freeVar);
        }
        return globalTrueNode;
    }// createTrueNode

public:
    /**
       Constructor of dDNNF compiler.

       @param[in] cnf, set of clauses
       @param[in] fWeights, the vector of literal's weight
       @param[in] c, true if the cache is activated, false otherwise
       @param[in] h, the variable heuristic name
       @param[in] p, the polarity phase heuristic name
       @param[in] _pv, the partitioner heuristic name
       @param[in] rp, true if we reverse the polarity, false otherwise
       @param[in] isProjectedVar, boolean vector used to decide if a variable is projected (true) or not (false)
    */
    DDnnfCompiler(vec<vec<Lit> > &cnf, vec<double> &wl, OptionManager &optList, vec<bool> &isProjectedVar,
                  ostream *certif) : s(certif)
    {
        isCertified = certif != NULL;
        for(int i = 0 ; i<wl.size()>>1 ; i++) s.newVar();
        for(int i = 0 ; i<cnf.size() ; i++) s.addClause_(cnf[i]);

        initUnsat = !s.solveWithAssumptions();

        if(!initUnsat)
        {
            s.simplify();
            s.remove_satisfied = false;
            s.setNeedModel(false);

            callPartitioner = callEquiv = 0;
            optCached = optList.optCache;
            optDecomposableAndNode = optList.optDecomposableAndNode;
            optReversePolarity = optList.reversePolarity;
            optList.printOptions();

            // initialized the data structure
            prepareVecClauses(clauses, s);
            occManager = new DynamicOccurrenceManager(clauses, s.nVars());

            freqLimitDyn = optList.freqLimitDyn;
            //cache = new CacheCNF<DAG<T> *>(optList.reduceCache, optList.strategyRedCache);
            cache = new CacheCNF<std::weak_ptr<DAG<T> > >(optList.reduceCache, optList.strategyRedCache);
            cache->initHashTable(occManager->getNbVariable(), occManager->getNbClause(),
                                 occManager->getMaxSizeClause());

            vs = new VariableHeuristicInterface(s, occManager, optList.varHeuristic,
                                                optList.phaseHeuristic, isProjectedVar);
            bm = new BucketManager<std::weak_ptr<DAG<T> > >(occManager, optList.strategyRedCache);
            pv = PartitionerInterface::getPartitioner(s, occManager, optList);

            alreadyAdd.initialize(s.nVars(), false);

            stampIdx = 0;
            stampVar.initialize(s.nVars(), 0);
            em.initEquivManager(s.nVars());

            globalTrueNode = std::make_shared<trueNode<T> >();
            globalFalseNode = std::make_shared<falseNode<T> >();

            // statistics initialization
            minAffectedAndNode = s.nVars();
            nbSplit = nbCallCompile = 0;
            currentTime = cpuTime();
            nbAndMinusNode = nbAndNode = nbDecisionNode = nbDomainConstraintNode = nbNodeInCompile = 0;
            sumAffectedAndNode = 0;
        }

        isProjectedVar.copyTo(DAG<T>::varProjected);
        wl.copyTo(DAG<T>::weights);
        for(int i = 0 ; i<s.nVars() ; i++) DAG<T>::weightsVar.push(wl[i<<1] + wl[(i<<1) | 1]);
        if (!initUnsat) cache->setInfoFormula(s.nVars(), cnf.size(), occManager->getMaxSizeClause());
    }// DDnnfCompiler


    ~DDnnfCompiler()
    {
        if(pv) delete pv;
        delete cache; delete vs; delete bm;
        delete occManager;
    }

    /**
       Compile the CNF formula into a dDNNF structure.

       \return a DAG
    */
    std::unique_ptr<rootNode<T> > compile()
    {
        std::shared_ptr<DAG<T> > d = nullptr;
        auto root = std::make_unique<rootNode<T> >(s.nVars());
        vec<Var> freeVariable, setOfVar, priorityVar;
        DAG<T>::initSizeVector(s.nVars());
        vec<int> idxReason;

        if(initUnsat) root->assignRootNode(s.trail, std::make_shared<falseNode<T> >(), false, s.nVars(), freeVariable, idxReason);
        else
        {
            bool fromCache = false;
            onTheBranch bData;
            if(!s.solveWithAssumptions()) d = globalFalseNode;
            else
            {
                for(int i = 0 ; i<s.nVars() ; i++) setOfVar.push(i);
                d = compile_(setOfVar, priorityVar, lit_Undef, bData, fromCache, idxReason);
            }

            assert(s.decisionLevel() == 0 && d);
            printFinalStatsCache();
            root->assignRootNode(bData.units, d, fromCache, s.nVars(), bData.free, idxReason);
        }
        return root;
    }// compile
};

#endif
