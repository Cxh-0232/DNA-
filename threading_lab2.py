import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from numba import njit

def kmer_match(s1, s2, k, m):
    """
    检查两个k-mer是否匹配，允许最多m个错配
    """
    if len(s1) != len(s2):
        return False
    mismatches = 0
    for a, b in zip(s1, s2):
        if a != b:
            mismatches += 1
            if mismatches > m:
                return False
    return True

def reverse_complement(seq):
    """
    计算DNA序列的反向互补序列
    """
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement[base] for base in reversed(seq))

def transform_diagonal_segments(matrix, min_length=1):
    """
    转换对角线线段，使用numpy优化
    """
    rows, cols = len(matrix), len(matrix[0])
    result = np.array(matrix, dtype=bool)
    
    # 只处理主对角线方向（左上到右下）
    direction = (1, 1)
    visited = np.zeros_like(matrix, dtype=bool)
    
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] and not visited[i][j]:
                # 找出对角线线段
                length = 1
                x, y = i + direction[0], j + direction[1]
                while 0 <= x < rows and 0 <= y < cols and matrix[x][y]:
                    visited[x][y] = True
                    length += 1
                    x += direction[0]
                    y += direction[1]
                
                if length >= min_length:
                    # 计算变换后的位置
                    a_r, a_c = i, j
                    b_r, b_c = x - direction[0], y - direction[1]
                    
                    # 清除原线段
                    x, y = a_r, a_c
                    for _ in range(length):
                        result[x][y] = False
                        x += direction[0]
                        y += direction[1]
                    
                    # 绘制新线段
                    new_a_r = rows - 1 - b_r
                    new_a_c = a_c
                    new_b_r = rows - 1 - a_r
                    new_b_c = b_c
                    
                    # 确定新线段方向
                    delta_r = 1 if new_b_r > new_a_r else -1
                    delta_c = 1 if new_b_c > new_a_c else -1
                    
                    x, y = new_a_r, new_a_c
                    for _ in range(length):
                        if 0 <= x < rows and 0 <= y < cols:
                            result[x][y] = True
                        x += delta_r
                        y += delta_c
    
    return result

def process_diagonal(d, ref_seq, que_seq, k, m, ref_len, que_len):
    """
    处理单个对角线，用于多线程
    """
    i_start = max(d, 0)
    j_start = max(-d, 0)
    diag_len = min(ref_len - i_start, que_len - j_start)
    
    matches = []
    i, j = i_start, j_start
    temp = 0
    
    while i < i_start + diag_len and j < que_len:
        current_k = min(k, i_start + diag_len - i)
        ref_kmer = ref_seq[i:i+current_k]
        que_kmer = que_seq[j:j+current_k]
        
        if current_k == k and kmer_match(ref_kmer, que_kmer, k, m):
            matches.extend((i + a, j + a) for a in range(current_k))
            i += k
            j += k
            temp = 1
        elif current_k == k and not kmer_match(ref_kmer, que_kmer, k, m):
            if temp == 1:
                while i < i_start + diag_len and j < que_len and ref_seq[i] == que_seq[j]:
                    matches.append((i, j))
                    i += 1
                    j += 1
                temp = 0
            else:
                i += 1
                j += 1
        elif current_k < k and ref_kmer == que_kmer:
            matches.extend((i + a, j + a) for a in range(current_k))
            i += current_k
            j += current_k
        else:
            if ref_seq[i] == que_seq[j] and temp == 1:
                matches.append((i, j))
            i += 1
            j += 1
    
    return matches

def build_complex_matrix(ref_seq, rev_ref_seq, que_seq, k, m, ref_pos, que_pos):
    """
    构建打分矩阵，使用多线程优化
    """
    ref_len, que_len = len(ref_seq), len(que_seq)
    score_matrix = np.zeros((ref_len, que_len), dtype=bool)
    
    # 使用线程池处理所有对角线
    with ThreadPoolExecutor() as executor:
        futures = []
        for d in range(-que_len + 1, ref_len):
            futures.append(executor.submit(
                process_diagonal, d, ref_seq, que_seq, k, m, ref_len, que_len
            ))
        
        for future in futures:
            for i, j in future.result():
                score_matrix[i, j] = True
    
    raw_matchings = extract_raw_matchings(score_matrix)
    
    # 反向匹配处理
    rev_matrix = np.zeros((ref_len, que_len), dtype=bool)
    with ThreadPoolExecutor() as executor:
        futures = []
        for d in range(-que_len + 1, ref_len):
            futures.append(executor.submit(
                process_diagonal, d, rev_ref_seq, que_seq, k, m, ref_len, que_len
            ))
        
        for future in futures:
            for i, j in future.result():
                rev_matrix[i, j] = True
    
    transformed_rev_matrix = transform_diagonal_segments(rev_matrix, 30)
    transformed_rev_matrix = np.array(transformed_rev_matrix, dtype=bool)
    raw_matchings += extract_raw_matchings(transformed_rev_matrix)
    raw_matchings.sort(key=lambda x: x[0])
    
    return raw_matchings

def extract_raw_matchings(matrix):
    """
    提取所有斜对角线上连续为1的线段，返回[(que_start, que_end, ref_start, ref_end), ...]
    """
    ref_len, que_len = matrix.shape
    result = []

    # 遍历所有可能的对角线（主对角线及其上下）
    for offset in range(-ref_len + 1, que_len):
        i = max(0, -offset)
        j = max(0, offset)
        ref_start = que_start = None
        length = 0
        while i < ref_len and j < que_len:
            if matrix[i, j]:
                if length == 0:
                    ref_start, que_start = i, j
                length += 1
            else:
                if length > 0:
                    if j - que_start >= 30:
                        result.append((que_start, j - 1, ref_start, i - 1))
                    length = 0
            i += 1
            j += 1
        if length > 0:
            if j - que_start >= 30:
                result.append((que_start, j - 1, ref_start, i - 1))
    return result

def plot_matchings_as_segments(raw_matchings, filename="ideal_condition.png"):
    """
    输入raw_matchings元组列表，绘制所有线段，并保存到文件
    每个元组格式为(que_start, que_end, ref_start, ref_end)
    """
    plt.figure(figsize=(15, 15), dpi=300)
    for que_start, que_end, ref_start, ref_end in raw_matchings:
        # 横轴为que，纵轴为ref，原点左上角
        plt.plot([que_start, que_end], [ref_start, ref_end], marker='o')
    plt.xlabel("Query (que_seq) 坐标")
    plt.ylabel("Reference (ref_seq) 坐标")
    plt.title("DNA序列比对线段图")
    plt.gca().invert_yaxis()  # 原点左上角
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"比对线段图已保存为: {filename}")

def calculate_distance(matchings, que_seq, ref_seq):
    total_distance = 0
    for que_start, que_end, ref_start, ref_end in matchings:
        for i in range(que_end - que_start + 1):
            if que_seq[que_start + i] != ref_seq[ref_start + i]:
                total_distance += 1
    return total_distance

def find_best_matchings(raw_matchings):
    """
    用图算法结合启发式规则找到最佳匹配情况，满足每个匹配长度大于等于30，且que在匹配中不重叠
    输入：匹配的所有节点
    """
    raw_matchings = [list(x) for x in raw_matchings]
    best_matchings = [raw_matchings[0]] # 初始状态，仅包含第一个匹配
    last_matching = raw_matchings[0]
    for matching in raw_matchings:
        if matching[1] > last_matching[1]:
            last_matching = matching

    while True: # 寻找下一状态
        if last_matching in best_matchings:
            break
        last_que_start = best_matchings[-1][0]
        last_que_end = best_matchings[-1][1]
        for matching in raw_matchings:
            que_start, que_end, ref_start, ref_end = matching
            if que_end <= last_que_end:
                continue
            elif que_start < last_que_end and que_end > last_que_end and (que_end - last_que_end) >= 30:
                best_matchings.append(matching)
                break
            elif que_start >= last_que_end and (que_end - last_que_end) >= 30:
                best_matchings.append(matching)
                break
            elif que_start < last_que_end and que_end > last_que_end and (que_end - last_que_start) >= 60 and (que_end - last_que_end) >= 25:
                best_matchings.append(matching)
                break
            else:
                continue
        
    # 删除重叠部分太多的匹配
    for i in range(len(best_matchings) - 2):
        #current_matchings = best_matchings[i:i+3]
        if i + 2 >= len(best_matchings):
            break
        first_matching, second_matching, third_matching = best_matchings[i], best_matchings[i + 1], best_matchings[i + 2]
        if third_matching[0] > first_matching[1] and third_matching[0] - first_matching[1] <= 5:
            best_matchings.pop(i + 1)

        # 确保que不重叠
    for i in range(len(best_matchings) - 1):
        cur_matching, next_matching = best_matchings[i], best_matchings[i + 1]
        if cur_matching[1] < next_matching[0]:
            cur_matching[1] += 1
            cur_matching[3] += 1
        elif cur_matching[1] > next_matching[0]:
            if cur_matching[1] - cur_matching[0] > next_matching[1] - next_matching[0]:
                cur_matching[1] = next_matching[0]
                cur_matching[3] = cur_matching[2] + cur_matching[1] - cur_matching[0]
            else:
                next_matching[0] = cur_matching[1]
                next_matching[2] = next_matching[3] + next_matching[0] - next_matching[1]
        else:
            continue

    print(f"best_matchings: {best_matchings}")
    return best_matchings

if __name__ == "__main__":
    # 示例序列
    ref_seq = 'TGATTTAGAACGGACTTAGCAGACATTGAAACTCGAGGGGTATAGCAATAGATGCCCAAAAAGGTAAGCGCCATAAGCGTGGTTCTACGAGCCAGGTGCTCATGCCTAAGTTCTGCGCCTTCGCTGTCACTTGGAAATACTGTAATGGATCATGCCTAGGTTATGCGCCTTCGGGGTCACTTCAACATACTGTAATGGATCATGCCTAGGTTTTGCGTGTTCGCTGTCATTTCGAAATACTCCAATGGATGATGCCTAGGTTCTGTGCCTTCGCTGACGCATGGAAATACTTTAACGGATCATGCCCAGGCTCTGCGCCTTCGCTGAAACTTCGAAATACTCTAATGGATCATGCCTCGGTGCTCCACCTTCGCTTTCATTCCGAAATACTCTAATGGATCGCGTCCGTGTAACAACTTCGTACTGTTATATCGACCATCAGAATACCCATCCCTCGGGGAGGTAACCTATATTCACGTCGCAAGTTTCGATCTACAGTATGCTGACTGTTTGCCGCGATTTTAAGTCAAGAAGCGCGTCCACATGGTCGCGGTCGTCAACTTCAGTACGCTCATATGACACCAAAAGATCTACCTACAGCCCGTGCAGCTCGACTTTTGTGCTCTAGGGCACGACGGGTGGCGTTTGCTCCCGCGCATCTCGACTTTTAAGCTCTATGGCACAACGTGTGGCGTTTGCCCCCGCGCAGCTCGACTTTTGTGCTCTAGGGCACGGCGGGTGGCGTTTGCCCTCGCCCAGCTTGACTTTTGTGCTCTAGGGCACGACGGGTGGCGTTTGCCCCCGTGCAGCCCGACTTTTGTACTCTAGTGCACGACGGGTGGCGTTTGCCCCCGCACCGCTCGACTTTTGTGATCTAGGGCACTACGAGTAGCGTTGGCCCAGACAGATCAACGCACATGGATTCTCTAACAGGCCCCGCGCTTCTCATTGGCCCGTGAGACGGGTCTGAGAGGAAGACATTAGGTAGATACGGAAAGCTTTGGCGTAGTTCGTATCTTTCAGCGGTGAAGCGTCTTCGGTCCGGGCTGCGTTATGCCTGCGGGAGGAAGGCTCCACTAGATGGTTTACGAGACATAATGTCAGCTTCCTAAAGGTACGGCAGCGCCTGCGTATATCACAGGACGAATTGTCAGTTTGCTAGGGGTACGGGAGCGCTTGCGTATTACATAGGACGAATCGTCAGCTTCCTAAAGGGACGGTAGCGCTTGCGTGTTACATAGGACGAATTGTCAGCTTCGTAAAGGTACGGTAGTTCTTGCGTATTACATAGGATGCATTGTCCGCTTCCTAAAGGTACGCTGGCGCTTGCGTATCACATAGGACGGATAGCGCGATTGCTAAAGGTACGGGAGCGCTTGCGTCTTAGAGCGCACGAATCGGATATAAGCTGCGCCCGCGTCTGGCGAGCAAAAATCGTGGGAGCCAGCGAGGGAAAAACTGCTCGGGCGACTTAAACGGAATTACAAGACTCATTGCCATCGAGGACGTTAGACTAAAGAGCCCCTGCGTGCCTCCTTTGTATAGCTCGATGTAGTGGCCCGTGTATGTGGAACAGGAATGCTCGATCTAAGGTAGTAGTGGCTACAGCTCCGAGAGTTTGCGTACTGCGGTGCCAGGGATTTTGCCTGCGGGTACAGCCTCTGCGCACGCCGGTCTGTGATCAAGAACTAAACTAGAGA'
    que_seq = 'TGATTTAGAACGGACTTAGCAGACATTGAAACTCGAGGGGTATAGCAATAGATGCCCAAAAAGGTAAGCGCCATAAGCGTGTTTCTACGAGCCAGGTGCTCATGCCTAAGTTCTGCGCCTTCGCTGTCACTGGGAAATACTGTAATGGATCATCCGTAGGTTATGCGCCTTCGGGGTCACTTCAACATACTGTAACGGATCGTGCCTAGGTTTTGCGTATTCGCTGTCATTTCGAATTACACCAATGGATGATGCCTAGGTTCTGTGCCTCCGCTGACGCATCGAAATACTTTAACGGATCGCGTCCGAGTAACAACTTCGTACTGTTATATAGGCAATCAGAATACCCATGCCTCGGGGAGGTAACCTATATTCACGTCGCAAGTTTCGATCTACAGTACTGTAGGTATATCTTTTGGTGTCATATGAGGGTACTGAACTTGACGACCGCGACCATGTGGATGCGCTTCTTGACTTAAAATCGCGGCAAACAGTAAGCATCCGTGAAGCTCGACTTTTGTGCTCTAGGGCACGACGGGTGGCGTTTGCTCCCGCGCATCTCGAGTTGTAAGCTCTATGGCACAACGGGTGGCGTTTGCCGCCGAGCAGCTCGACTTTTGTGCTCTAGGGCACGGCGGGTGGCGTTTGCCCTCGCCCAGCTTGACTTTTGTGCTCTAGGGCACGACGGGTGGCCTTTGCCCCCGCGCAGCTGGACTTTTGTGCTCTAGGGCACGGCGGGTGGCGTTTGCCCTCCCCCAGCTTGACTACTGTGCTCTAGGGCACGACGGGTGGCGTTTGCCCCCGCGCAGCTCGACTTTTGTGCTCTATGGCACGGGGGGTGGCGTTTGCCCTCGCCCAGCTTGACTTTTGCGCTCTAGGGCACGACGGGTGGCGTTTGCCGGCAAACGCCACACGTCGTGCCCTAGAGCACAAACGTCAAGCTGGGCGAGGGCAACCGCCACCCGCCCTGCCCTAGAGCACAAAAGTCGAGCTGCGCGGGCCCGCGCAGCTCGACTTTTGTGCTCTAGGACACGGCGGGTGGCGTTTGCCCTCGCCCAGCTTGACTCTTGTGCTCTAGGGCACGACGGGTGGCGTTTGCCCCAGCGCAGCCCGACTTTTGTACTCTAGAGCACGACGGTTGGCATTTGCCCCCGCACCGCTCGACTTTTGTGATCTAGGGCCCTAGGAGTAGCGTTGGCCAGCTTTCCGTATCTACCTAATGTCTTCCTCTCAGACCCGTCTCACGGGCCAATGAGAAGCGCGGGGCCTATTAGAGAATCCATGTGCGTTGATCTGTCTGCAGACAGCTCAACGCACATGGATTCGCTAGCAGGCCCCGCGCTTCTCATTGGCCCGTGAGACGGGTCTGAGAGGAAGACATAAGGTAGATACGGCAAGCTCACGTCCGTGTAACAACGTCGTACTGTTATATCGACCATCAGAATCCCCATCCCGCGAGGAGGTAACCTATATTCAGGTCGCAAGTTTCGATCTACAGTATTGGCGTAGTTCGTATCTTTCAGCGGTGAAGCTTCTTCGGTCCGGGCTGCGTTATGCCTGCTGGAGGACGGCTCCACTAGATGGTTTACGAGACATAATGTCCGCTTCCTAAAGGTACACTGGCGCTTGAGTATCACATAGGACGGATAGCTCGATTCCTAAAGGGACGGGAGCGCTTGCGTCTTAGAGCGCATGAATCGTCAGCTTCCCAAAGGGACCGTAGCGCTTGCGTGTTATATAGGAAGAATGGTCAGCTTTGTAAAGGTACGGTAGTTCTTGCGTATTACAGAGGATGCATTGTCTACTACCTAAAGGTACGGCAGCGCCTGCGTATATCACAGGACGAATTGTCAGTTTGCTAGGGGTACGGGAGCGCTTGCATATTACATAGGACGAATCGGATATAAGCTGCGCCCGCGTCTGGCGATAAAAAATCGTGGTAGCCAGCGAGGGAAAAACTGCTCGGGCGACTTAAACGGAATTAAAAGACTCATTGCCGTGACAGACTTCCGTATAGCAACCTCTGGGATGTCGATGCGGTGTCCCCAGTCTGCGCTGAGCGGGGGCAGACAGACTTAGTTATAGTATGCATCTGTTTTAGCTAGACATCACGACCTAGTGGGGTTCATGTTGAGATTCTAGGGCGGTACGCAGCCGGTGGATTATTACTTCCCCAGAAATTCTGACTTCGTCACTGGATGGATTGTACTATCCGGTCAACCTTACAAGGTTTCAACAGGGACGAAGGGTAAACGTATGAAGCTTGGATGCCGTTACCGTAAAGGGCCCTATTGAAGTGTCGAGGACGTTAGACTAAAGAGCCCCTGCGTGCCTCCTTTGTATAGCTCGAGGTAGTGGCCCGGATATGTGGAACAGGAATGCTCGATCTAAGGTAGTAGTGGGTACCGCTCCGAGAGTTTGCGTACTGCGGTGCCCGGGATTTTGCCTGCGGGTACAGCCTCTGCGCACGCCGGTCTGTAATCAAGAACTAAACTAGAGA'
    
    k = 10 # kmer_size
    m = 1 # 允许1个错配
    
    rev_ref_seq = reverse_complement(ref_seq)
    raw_matchings = build_complex_matrix(ref_seq, rev_ref_seq, que_seq, k, m, 0, 0)

    print(f"\nk-mer大小: {k}, 最大错配: {m}")
    print(raw_matchings)

    best_matchings = find_best_matchings(raw_matchings)

    plot_matchings_as_segments(raw_matchings, "raw_matchings.png")
    plot_matchings_as_segments(best_matchings, "best_matchings.png")