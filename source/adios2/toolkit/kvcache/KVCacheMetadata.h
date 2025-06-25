//
// Created by chang on 3/20/25.
//

#ifndef ADIOS2_KVCACHEMETADATA_H
#define ADIOS2_KVCACHEMETADATA_H

#include "QueryBox.h"
#include "KVCacheCommon.h"

#ifdef ADIOS2_HAVE_SPATIALINDEX
#include <spatialindex/SpatialIndex.h>
using namespace SpatialIndex;
#endif

namespace adios2
{

namespace kvcache
{

#ifdef ADIOS2_HAVE_SPATIALINDEX
class MyVisitor : public IVisitor
{
public:
    MyVisitor() = default;

    std::vector<const IShape*> results;  // Store overlapping regions

    void visitNode(const INode& n) override{
        // Do nothing
    }

    // Called when visiting data (leaf entries)
    void visitData(const IData& d) override {
        IShape* shape = nullptr;
        d.getShape(&shape);  // Get the region (shape) of the data entry
        results.push_back(shape);  // Store overlapping region
    }

    void visitData(std::vector<const IData*>& v) override {
        for (const IData* data : v) {
            visitData(*data);
        }
    }
};
#endif

class KVCacheMetadata
{
private:
    int64_t indexIdentifier = 0;
#ifdef ADIOS2_HAVE_SPATIALINDEX
public:
    SpatialIndex::ISpatialIndex* m_tree = nullptr;
    size_t m_dim;

    KVCacheMetadata() = default;

    ~KVCacheMetadata() {
        if (m_tree != nullptr) {
            std::cout << *m_tree << std::endl;
            delete m_tree;
        }
    }

    void CreateNewTree(size_t capacity) {
        IStorageManager* memoryFile = StorageManager::createNewMemoryStorageManager();
        m_tree = SpatialIndex::RTree::createNewRTree(*memoryFile, 0.7, capacity, capacity, m_dim, SpatialIndex::RTree::RV_RSTAR, indexIdentifier);
    }

    void Insert(const QueryBox &queryBox) {
        double* pLow = new double[m_dim];
        double* pHigh = new double[m_dim];

        for (size_t i = 0; i < m_dim; ++i) {
            pLow[i] = queryBox.Start[i];
            pHigh[i] = queryBox.Start[i] + queryBox.Count[i];
        }

        SpatialIndex::Region r(pLow, pHigh, m_dim);
        m_tree->insertData(0, nullptr, r, indexIdentifier++);
    }

    void Query(const QueryBox &queryBox, const size_t &max_depth, size_t current_depth,
               std::vector<QueryBox> &regularBoxes, std::vector<QueryBox> &cachedBoxes,
                 KVCacheCommon kvcache, const std::string &keyPrefix) {
        if (current_depth > max_depth)
        {
            return;
        }
        current_depth++;

        double* pLow = new double[m_dim];
        double* pHigh = new double[m_dim];

        for (size_t i = 0; i < m_dim; ++i) {
            pLow[i] = queryBox.Start[i];
            pHigh[i] = queryBox.Start[i] + queryBox.Count[i];
        }

        MyVisitor visitor;
        SpatialIndex::Region r(pLow, pHigh, m_dim);
        m_tree->intersectsWithQuery(r, visitor);

        QueryBox maxOverlapBox = QueryBox(m_dim);
        QueryBox maxInteractBox = QueryBox(m_dim);
        for (const IShape* shape : visitor.results) {
            const SpatialIndex::Region* region = dynamic_cast<const SpatialIndex::Region*>(shape);
            QueryBox overlapBox = QueryBox(m_dim);
            for (size_t i = 0; i < m_dim; ++i) {
                overlapBox.Start[i] = region->m_pLow[i];
                overlapBox.Count[i] = region->m_pHigh[i] - region->m_pLow[i];
            }
            QueryBox intersectionBox = QueryBox(m_dim);
            overlapBox.IsInteracted(queryBox, intersectionBox);

            if (maxInteractBox.size() < intersectionBox.size()) {
                // check if maxDepth (scheduler) is set and intersectionBox size is larger than 1 MB / 8 B (double)
                if (max_depth != 999 && intersectionBox.size() < 1024 * 1 / sizeof(double)) {
                    continue;
                }
                // check if the box is still cached (not evicted)
                std::string key = keyPrefix + overlapBox.toString();
                if (!kvcache.Exists(key)) {
                    std::cout << "Key has been evicted: " << key << std::endl;
                    continue;
                }
                maxOverlapBox = overlapBox;
                maxInteractBox = intersectionBox;
                if (maxInteractBox.size() == queryBox.size()) {
                    break;
                }
            }
        }
        
        if (maxInteractBox.size() == 0) {
            regularBoxes.push_back(queryBox);
            return;
        }

        cachedBoxes.push_back(maxOverlapBox);
        std::cout << "[Partial] Going to retrieve " << maxOverlapBox.toString() << " from cache" << std::endl;
        std::cout << "[Partial] Going to reuse " << maxInteractBox.toString() << " from cache" << std::endl;

        if (maxInteractBox.size() == queryBox.size()) {
            return;
        }

        if (current_depth == max_depth) {
            maxInteractBox.NdCut(queryBox, regularBoxes);
        } else {
            std::vector<QueryBox> nextBoxes;
            maxInteractBox.NdCut(queryBox, nextBoxes);
            for (QueryBox &nextBox : nextBoxes) {
                this->Query(nextBox, max_depth, current_depth, regularBoxes, cachedBoxes, kvcache, keyPrefix);
            }
        }
    }

    void PrintTree() {
        std::cout << *m_tree << std::endl;
    }

#else
public:
    void* m_tree = nullptr;
    size_t m_dim = 0;
    KVCacheMetadata() = default;
    ~KVCacheMetadata() {}

    void CreateNewTree(size_t capacity) {}
    void Insert(const QueryBox &queryBox) {}
    void Query(const QueryBox &queryBox, const size_t &max_depth, size_t current_depth,
               std::vector<QueryBox> &regularBoxes, std::vector<QueryBox> &cachedBoxes,
               KVCacheCommon kvcache, const std::string &keyPrefix) {}
    void PrintTree() {}
#endif

};
}; // namespace kvcache
}; // adios2
#endif // ADIOS2_KVCACHEMETADATA_H
