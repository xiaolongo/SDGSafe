from collections import Counter

import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn
from torch_geometric.utils import coalesce


def classify_indices_vectorized(indices, masks, mask_names):
    """
    向量化地根据多个mask对索引进行分类
    """
    # 将索引转换为mask的布尔矩阵
    mask_matrix = torch.stack(masks, dim=1)

    # 获取目标索引对应的mask状态
    target_masks = mask_matrix[indices]

    # 找到每个索引对应的第一个为True的mask
    mask_indices = torch.argmax(target_masks.int(), dim=1)

    results = []
    for i, idx in enumerate(indices):
        mask_idx = mask_indices[i]
        if target_masks[i, mask_idx]:  # 确保确实为True
            results.append({mask_names[mask_idx]: int(idx)})
        else:
            results.append({"unknown": int(idx)})

    return results


def get_most_frequent_non_uncertain(data):
    """
    找出出现次数最多的非'uncertain'键及其对应的元素
    """
    keys = [list(d.keys())[0] for d in data]
    filtered_keys = [key for key in keys if key != "uncertain"]

    if filtered_keys:
        most_common_key = Counter(filtered_keys).most_common(1)[0][0]
        result_element = [item for item in data if most_common_key in item]
    else:
        raise ValueError("没有非'uncertain'的元素")
    return result_element


def get_uncertain(data):
    keys = [list(d.keys())[0] for d in data]
    filtered_keys = [key for key in keys if key == "uncertain"]
    if filtered_keys:
        most_common_key = Counter(filtered_keys).most_common(1)[0][0]
        result_element = [item for item in data if most_common_key in item]
    else:
        raise ValueError("没有非'uncertain'的元素")
    return result_element


def get_masked_edge_index(data, knn_k, uncertain_mask, ind_mask, ood_mask):
    """
    根据不确定节点的邻居节点的掩码分类，重新分配邻居节点索引
    """
    masks = [uncertain_mask, ind_mask, ood_mask]
    mask_names = ["uncertain", "ind", "ood"]

    assign_index = knn(data.x, data.x, knn_k)
    print(assign_index)

    revised_index_list = []

    for i in range(data.num_nodes):
        if not uncertain_mask[i]:
            continue
        column_indices = torch.nonzero(assign_index[0] == i, as_tuple=True)[0]
        current_neighbor_node = assign_index[1][column_indices]
        print("当前邻域节点:", current_neighbor_node)

        classification = classify_indices_vectorized(
            current_neighbor_node, masks, mask_names
        )
        print("向量化分类结果:", classification)

        most_common = get_most_frequent_non_uncertain(classification)
        most_common_neighbor = [v for d in most_common for _, v in d.items()]
        print(most_common_neighbor)
        uncertain = get_uncertain(classification)
        print(uncertain)
        most_common = most_common + uncertain
        print(most_common)

        indices = [
            current_neighbor_node.tolist().index(item) for item in most_common_neighbor
        ]
        print(indices)
        indices = list(map(lambda x: x + i * current_neighbor_node.size()[0], indices))
        print(indices)
        revised_index_list.append(assign_index[:, indices])
        break
    revised_index = torch.cat(revised_index_list, dim=1)

    return revised_index


def get_masked_edge_index_str(data, knn_k, uncertain_mask, ind_mask, ood_mask):
    """
    根据不确定节点的邻居节点的掩码分类，重新分配邻居节点索引
    """
    masks = [uncertain_mask, ind_mask, ood_mask]
    mask_names = ["uncertain", "ind", "ood"]

    assign_index = data.edge_index
    print(assign_index)

    # revised_index = torch.zeros_like(assign_index)
    # revised_index[0] = assign_index[0]
    revised_index_list = []
    sum_of_neighbors = 0  # 统计邻域节点累加值

    for i in range(data.num_nodes):

        # 获取当前节点的原始邻域
        column_indices = torch.nonzero(assign_index[0] == i, as_tuple=True)[0]
        current_neighbor_node = assign_index[1][column_indices]
        # print("节点原始邻域：", current_neighbor_node)

        sum_of_neighbors += current_neighbor_node.size()[0]

        classification = classify_indices_vectorized(
            current_neighbor_node, masks, mask_names
        )
        print("原始邻域分类结果:", classification)

        if ind_mask[i]:
            ind_values = [item["ind"] for item in classification if "ind" in item]
            ind_indices = [
                current_neighbor_node.tolist().index(item) for item in ind_values
            ]
            ind_indices = list(
                map(
                    lambda x: x + sum_of_neighbors - current_neighbor_node.size()[0],
                    ind_indices,
                )
            )
            revised_index_list.append(assign_index[:, ind_indices])
            print(f"第{i}个点:", assign_index[:, ind_indices])

        elif ood_mask[i]:
            ood_values = [item["ood"] for item in classification if "ood" in item]
            ood_indices = [
                current_neighbor_node.tolist().index(item) for item in ood_values
            ]
            ood_indices = list(
                map(
                    lambda x: x + sum_of_neighbors - current_neighbor_node.size()[0],
                    ood_indices,
                )
            )
            revised_index_list.append(assign_index[:, ood_indices])
            print(f"第{i}个点:", assign_index[:, ood_indices])

        else:  # for uncertain node
            assign_index_knn = knn(data.x, data.x, knn_k)  #
            column_indices_knn = torch.nonzero(assign_index_knn[0] == i, as_tuple=True)[
                0
            ]
            current_neighbor_node_knn = assign_index_knn[1][column_indices_knn]
            classification_knn = classify_indices_vectorized(
                current_neighbor_node_knn, masks, mask_names
            )
            most_common_knn = get_most_frequent_non_uncertain(classification_knn)
            most_common_knn = next(iter(most_common_knn[0].keys()))
            print("最多的邻域节点名:", most_common_knn)  # ind

            add_values = [
                item[most_common_knn]  # ood or ind
                for item in classification
                if most_common_knn in item
            ]
            add_uncertain_values = [
                item["uncertain"] for item in classification if "uncertain" in item
            ]
            add_values += add_uncertain_values

            add_indices = [
                current_neighbor_node.tolist().index(item) for item in add_values
            ]

            add_indices = list(
                map(
                    lambda x: x + sum_of_neighbors - current_neighbor_node.size()[0],
                    add_indices,
                )
            )
            revised_index_list.append(assign_index[:, add_indices])
            print(f"第{i}个点:", assign_index[:, add_indices])
        # break
    revised_index = torch.cat(revised_index_list, dim=1)
    revised_index = coalesce(revised_index)

    return revised_index


def get_ada_edge_index(data, uncertain_mask, ind_mask, ood_mask):
    masks = [uncertain_mask, ind_mask, ood_mask]
    mask_names = ["uncertain", "ind", "ood"]

    assign_index = data.edge_index
    row, _ = assign_index
    value = torch.ones_like(row)
    sum_of_neighbors = 0

    for i in range(data.num_nodes):
        # 获取当前节点的原始邻域
        column_indices = torch.nonzero(assign_index[0] == i, as_tuple=True)[0]
        current_neighbor_node = assign_index[1][column_indices]
        print("节点原始邻域：", current_neighbor_node)

        value_i = torch.ones_like(current_neighbor_node)
        sum_of_neighbors += current_neighbor_node.size()[0]

        classification = classify_indices_vectorized(
            current_neighbor_node, masks, mask_names
        )
        print("原始邻域分类结果:", classification)

        if ind_mask[i]:
            # 获取ood的索引，并把value赋值-1
            ood_values = [item["ood"] for item in classification if "ood" in item]
            if ood_values != []:
                index_map = {
                    value.item(): idx for idx, value in enumerate(current_neighbor_node)
                }
                indices = [index_map[item] for item in ood_values if item in index_map]
                print(indices)
                value_i[indices] = -1
                value[
                    sum_of_neighbors
                    - current_neighbor_node.size()[0] : sum_of_neighbors,
                ] = value_i
                print(value)
        if ood_mask[i]:
            # 获取ood的索引，并把value赋值-1
            ind_values = [item["ind"] for item in classification if "ind" in item]
            if ind_values != []:
                index_map = {
                    value.item(): idx for idx, value in enumerate(current_neighbor_node)
                }
                indices = [index_map[item] for item in ind_values if item in index_map]
                print(indices)
                value_i[indices] = -1
                value[
                    sum_of_neighbors
                    - current_neighbor_node.size()[0] : sum_of_neighbors,
                ] = value_i
                print(value)
    return value


x = torch.randn(6, 5)  # 6个节点，每个节点16维特征
e = torch.randn(6, 1)  # 能量值
edge_index = torch.tensor(
    [
        [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        [1, 0, 2, 3, 5, 2, 4, 4, 1, 0, 2, 1, 2, 3, 4],
    ],
    dtype=torch.int64,
)
# print(edge_index)
data = Data(x=x, edge_index=edge_index)

uncertain_mask = torch.tensor([False, False, True, False, True, False])
ind_mask = torch.tensor([True, True, False, True, False, False])
ood_mask = torch.tensor([False, False, False, False, False, True])

value = get_ada_edge_index(data)
print(value)

# revised_edge_index = get_masked_edge_index_str(
#     data, knn_k=4, uncertain_mask=uncertain_mask, ind_mask=ind_mask, ood_mask=ood_mask
# )
# # row, col = revised_edge_index
# print(revised_edge_index)
