/*!
 * Copyright by Contributors 2017
 * modified by baotong 2017.10
 */
#ifndef XGBOOST_GBTREE_MODEL_H
#define XGBOOST_GBTREE_MODEL_H

#include <utility>
#include <string>
#include <vector>
#include <fstream>
#include "tree_model.h"

namespace xgboost {
    namespace gbm {
/*! \brief model parameters */
        //struct GBTreeModelParam : public dmlc::Parameter<GBTreeModelParam> {
        struct GBTreeModelParam {
            /*! \brief number of trees */
            int num_trees;
            /*! \brief number of roots */
            int num_roots;
            /*! \brief number of features to be used by trees */
            int num_feature;
            /*! \brief pad this space, for backward compatibility reason.*/
            int pad_32bit;
            /*! \brief deprecated padding space. */
            int64_t num_pbuffer_deprecated;
            /*!
             * \brief how many output group a single instance can produce
             *  this affects the behavior of number of output we have:
             *    suppose we have n instance and k group, output will be k * n
             */
            int num_output_group;
            /*! \brief size of leaf vector needed in tree */
            int size_leaf_vector;
            /*! \brief reserved parameters */
            int reserved[32];

            /*! \brief constructor */
            GBTreeModelParam() {
                std::memset(this, 0, sizeof(GBTreeModelParam));
                static_assert(sizeof(GBTreeModelParam) == (4 + 2 + 2 + 32) * sizeof(int),
                              "64/32 bit compatibility issue");
            }
        };

        class GBTreeModel {
        public:
            explicit GBTreeModel(bst_float base_margin) : base_margin(base_margin) {}

            void Configure(const std::vector <std::pair<std::string, std::string> > &cfg) {
                // initialize model parameters if not yet been initialized.
                if (trees.size() == 0) {
					// TODO: init
                    //param.InitAllowUnknown(cfg);
                }
            }

            void InitTreesToUpdate() {
                trees.clear();
                param.num_trees = 0;
                tree_info.clear();
            }

            void Load(std::ifstream& ifile) {
                if (!ifile.read((char*)&param, sizeof(param))) {
                    std::cerr << "GBTree:: invalid model file" << std::endl;
                }

                trees.clear();

                for (int i = 0; i < param.num_trees; ++i) {
                    std::unique_ptr<RegTree> ptr(new RegTree());
                    ptr->Load(ifile);
                    trees.push_back(std::move(ptr));
                }
                
                tree_info.resize(param.num_trees);
                if (param.num_trees != 0) {
                    ifile.read((char*)&tree_info[0], sizeof(int) * param.num_trees);
                }
                
            }
            
            inline float PredictInstanceRaw(FVec &feats, unsigned tree_begin, unsigned tree_end) {
                // bst_float psum = this->base_margin;
                bst_float psum = 0.0f;
                for (size_t i = tree_begin; i < tree_end; ++i) {
                    // bst_group = 1, for binary classification
                    // default root_index=0
                    psum += trees[i]->Predict(feats);
                    
                }
                return psum;
            }

            inline bool getLeafOnehotSparse(FVec &feats, 
                        std::vector<int>& onehot_vec, unsigned tree_begin, unsigned tree_end) { 
                // std::cout << "transform to sparse" << std::endl;
                int offset = 0, ind = 0, tree_num = tree_end-tree_begin;
                onehot_vec.resize(tree_num, -1);
                while ( ind < tree_num ) {
                    std::pair<int, int> onehot_i = 
                        trees[ind+tree_begin]->GetLeaf1HotEncode(feats);
                    int leaf_i = onehot_i.first, leaf_num_i = onehot_i.second;
                    // std::cout << "tree[" << ind+tree_begin << "], leaf index:" << leaf_i << ", leaves num:" << leaf_num_i << std::endl;
                    if( leaf_i == -1 || leaf_i >= leaf_num_i )
                        return false;
                    onehot_vec[ind] = leaf_i + offset;
                    ++ ind;
                    offset += leaf_num_i;
                }
                return true;
            }

            inline bool getLeafOnehotDense(FVec &feats, 
                        std::vector<int>& onehot_vec, unsigned tree_begin, unsigned tree_end) { 
                onehot_vec.clear();
                int total_leaves = 0;
                for (size_t i = tree_begin; i < tree_end; ++i) {
                    std::pair<int, int> onehot_i = trees[i]->GetLeaf1HotEncode(feats);
                    int leaf_i = onehot_i.first, leaf_num_i = onehot_i.second;
                    // std::cout << "tree[" << i << "], leaf index:" << leaf_i << ", leaves num:" << leaf_num_i << std::endl;
                    if( leaf_i == -1 || leaf_i >= leaf_num_i )
                        return false;
                    std::vector<int> tmp(leaf_num_i, 0);
                    tmp[leaf_i] = 1;
                    total_leaves += leaf_num_i;
                    onehot_vec.insert(onehot_vec.end(), tmp.begin(), tmp.end());
                }
                onehot_vec.resize( total_leaves );
                return true;
            }
            
        public:
            // base margin
            bst_float base_margin;
            // model parameter
            GBTreeModelParam param;
            /*! \brief vector of trees stored in the model */
            std::vector <std::unique_ptr<RegTree> > trees;
            /*! \brief for the update process, a place to keep the initial trees */
            //std::vector<std::unique_ptr<RegTree> > trees_to_update;
            /*! \brief some information indicator of the tree, reserved */
            std::vector<int> tree_info;
        };
    }  // namespace gbm
}  // namespace xgboost

#endif
