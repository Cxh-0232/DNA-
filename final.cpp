#include<iostream>
#include<string>
#include<vector>
#include<cstring>
#include<algorithm>
#include<queue>
#include<unordered_map>
#include<iomanip>

struct align_matrix {
    int **ptr;
    int rows, cols;
    align_matrix(int rows, int cols) : rows(rows), cols(cols) {
        ptr = new int*[rows];
        for(int row = 0; row < rows; row++) {
            ptr[row] = new int[cols];
            for(int col = 0; col < cols; col++) {
                ptr[row][col] = 0;
            }
        }
    }
    int* operator[](int index) const {
        return ptr[index];
    }
};

struct repetition_info {
    repetition_info() {
        fragment = '\0';
        pos = 0;
        length = 0;
        times = 0;
        is_reversed = false;
    }
    std::string fragment;
    int pos;
    int length;
    int times;
    bool is_reversed;
};

std::vector<std::vector<int>> reverseDiagonals(std::vector<std::vector<int>>& matrix) {
    if (matrix.empty() || matrix[0].empty()) {
        return matrix;
    }
    int rows = matrix.size();
    int cols = matrix[0].size();
    for (int diff = -(cols - 1); diff <= rows - 1; ++diff) {
        std::vector<std::pair<int, int>> currentSeqPositions;
        std::vector<int> currentSeqElements;
        for (int i = 0; i < rows; ++i) {
            int j = i - diff;
            if (j >= 0 && j < cols) {
                int val = matrix[i][j];
                if (val == 0) {
                    if (!currentSeqElements.empty()) {
                        reverse(currentSeqElements.begin(), currentSeqElements.end());
                        for (size_t k = 0; k < currentSeqElements.size(); ++k) {
                            int row = currentSeqPositions[k].first;
                            int col = currentSeqPositions[k].second;
                            matrix[row][col] = currentSeqElements[k];
                        }
                        currentSeqElements.clear();
                        currentSeqPositions.clear();
                    }
                } else {
                    if (currentSeqElements.empty()) {
                        if (val == 1) {
                            currentSeqElements.push_back(val);
                            currentSeqPositions.emplace_back(i, j);
                        }
                    } else {
                        if (val == currentSeqElements.back() + 1) {
                            currentSeqElements.push_back(val);
                            currentSeqPositions.emplace_back(i, j);
                        } else {
                            reverse(currentSeqElements.begin(), currentSeqElements.end());
                            for (size_t k = 0; k < currentSeqElements.size(); ++k) {
                                int row = currentSeqPositions[k].first;
                                int col = currentSeqPositions[k].second;
                                matrix[row][col] = currentSeqElements[k];
                            }
                            currentSeqElements.clear();
                            currentSeqPositions.clear();
                            if (val == 1) {
                                currentSeqElements.push_back(val);
                                currentSeqPositions.emplace_back(i, j);
                            }
                        }
                    }
                }
            }
        }
        if (!currentSeqElements.empty()) {
            reverse(currentSeqElements.begin(), currentSeqElements.end());
            for (size_t k = 0; k < currentSeqElements.size(); ++k) {
                int row = currentSeqPositions[k].first;
                int col = currentSeqPositions[k].second;
                matrix[row][col] = currentSeqElements[k];
            }
        }
    }
    return matrix;
}

class dag {
//private:
public:
    std::vector<std::vector<int>> list;
public:
    void addNode(int node) {
        if (node >= list.size()) {
            list.resize(node + 1);
        }
        if(list[node].empty()) list[node].push_back(node);
    }
    void addEdge(int from, int to) {
        if(!list[from].empty() && !list[to].empty()) list[from].push_back(to);
    }
    std::vector<int> findShortestPath() {
        if (list.empty()) return {};
        int min_node = list[0][0];
        int max_node = list.back()[0];
        std::unordered_map<int, std::vector<int>> graph;
        for (const auto& node_list : list) {
            if (node_list.empty()) continue;
            int node = node_list[0];
            for (size_t i = 1; i < node_list.size(); ++i) {
                graph[node].push_back(node_list[i]);
            }
        }
        std::queue<std::vector<int>> q;
        q.push({min_node});
        while (!q.empty()) {
            std::vector<int> path = q.front();
            q.pop();
            int current = path.back();
            if (current == max_node) {
                return path;
            }
            for (int neighbor : graph[current]) {
                std::vector<int> new_path = path;
                new_path.push_back(neighbor);
                q.push(new_path);
            }
        }
        return {};
    }
};

class alignment {
public:
    alignment(const std::string& ref, const std::string& query)
    : Reference(ref), Query(query), ref_len(ref.length()), que_len(query.length()) {
        forward.assign(ref_len, std::vector<int>(que_len, 0));
        reverse.assign(ref_len, std::vector<int>(que_len, 0));
    }
    ~alignment(){}
    std::string get_ref() const {return Reference;}
    std::string get_que() const {return Query;}
    std::vector<repetition_info> find_repetition();
    std::vector<int> make_dag(int que_start) const;
private:
    const std::string Reference, Query;
    int ref_len, que_len;
    std::vector<std::vector<int>> forward, reverse;
};

std::vector<repetition_info> alignment::find_repetition() {
    std::vector<repetition_info> results;
    int r_start = -1, que_start; //row,col
    // forward matrix
    for(int i = 0; i < que_len; i++) {
        if(Query[i] == Reference[0]) forward[0][i] = 1;
    }
    for(int i = 1; i < ref_len; i++) {
        for(int j = i; j < que_len; j++) {
            if(Query[j] == Reference[i]) {
                forward[i][j] = forward[i - 1][j - 1] + 1;
            }
        }
    }
    for(int i = 1; i < ref_len; i++) {
        if(forward[i][i] == 0) {
            r_start = i - 1;
            break;
        }
    }
    // reverse matrx
    std::string reversed_ref = Reference;
    std::reverse(reversed_ref.begin(), reversed_ref.end());
    for(int i = 0; i < que_len; i++) {
        switch(reversed_ref[i]) {
            case 'A': {
                reversed_ref[i] = 'T';
                break;
            } case 'C': {
                reversed_ref[i] = 'G';
                break;
            } case 'G': {
                reversed_ref[i] = 'C';
                break;
            } case 'T': {
                reversed_ref[i] = 'A';
                break;
            } default: {}
        }
    } 
    for(int i = 0; i < que_len; i++) {
        if(reversed_ref[0] == Query[i]) reverse[0][i] = 1;
    }
    for(int i = 1; i < ref_len; i++) {
        for(int j = i; j < que_len; j++) {
            if(reversed_ref[i] == Query[j]) {
                reverse[i][j] = reverse[i - 1][j - 1] + 1;
            }
        }
    }
    reverse = reverseDiagonals(reverse);
    que_start = r_start + 1;
    std::vector<int> dag_visited;
    while(1) {
        dag_visited = this->make_dag(que_start);
        int len = dag_visited.size();
        if (len == 0 || dag_visited.back() != (que_len - (ref_len - r_start))) {
            que_start--;
            r_start--;
        } else {
            break;
        }
    }
    for(int i = 0; i < dag_visited.size() - 1; i++) {
        repetition_info rep;
        rep.pos = dag_visited[0] + 1;
        rep.length = dag_visited[i + 1] - dag_visited[i];
        rep.times = 1;
        rep.fragment = Query.substr(dag_visited[i] + 1, rep.length);
        if(rep.fragment == Reference.substr(rep.pos - rep.length, rep.length)) rep.is_reversed = false;
        else rep.is_reversed = true;
        int repeated = 0;
        for(int i = 0; i < results.size(); i++) {
            if(rep.pos == results[i].pos && rep.length == results[i].length && rep.is_reversed == results[i].is_reversed) {
                results[i].times++;
                repeated = 1;
                break;
            }
        }
        if(repeated == 0) results.push_back(rep);
    }
    return results;
}

std::vector<int> alignment::make_dag(int que_start) const {
    int r_start = que_start - 1;
    std::vector<std::vector<int>> result;
    std::vector<int> points;
    dag dag;
    dag.addNode(r_start);
    for(int i = que_start; i < (que_len - (ref_len - r_start - 1)); i++) {
        if(reverse[ref_len - r_start - 1][i] > 0) {
            int node = i + reverse[ref_len - r_start - 1][i] - 1;
            dag.addNode(i - 1);
            if(node < (que_len - (ref_len - r_start - 1))) dag.addNode(node);
            else dag.addNode(que_len - (ref_len - r_start));
        }
    }
    for(int i = que_start; i < (que_len - (ref_len - r_start - 1)); i++) {
        for(int j = r_start; j < i; j++) {
            if(forward[r_start][i] >= i - j) {
                dag.addNode(i);
                dag.addEdge(j, i);
            }
        }
    }
    for(int i = que_start; i < (que_len - (ref_len - r_start - 1)); i++) {
        for(int j = i + 1; j < (que_len - (ref_len - r_start - 1)); j++) {
            if(reverse[ref_len - r_start - 1][i] >= j - i + 1) {
                dag.addEdge(i - 1, j);
            }
        }
    }
    std::vector<std::vector<int>> copy_list;
    for(int i = 0; i < dag.list.size(); i++) {
        if(dag.list[i].empty()) continue;
        copy_list.push_back(dag.list[i]); 
    }
    dag.list = copy_list;
    std::vector<int> visited = dag.findShortestPath();
    return visited;
}

int main() {
    std::string ref, query;
    std::cin >> ref >> query;
    alignment require(ref, query);
    std::vector<repetition_info> results; 
    results = require.find_repetition();
    int idWidth = 6, seqWidth = 100, posWidth = 16, lenWidth = 15, timesWidth = 14, reversedWidth = 11;
    std::cout << "| " << std::left << std::setw(idWidth) << "number" << " | "
              << std::setw(seqWidth) << "repeated segment" << " | "
              << std::setw(posWidth) << "pos in reference" << " | "
              << std::setw(lenWidth) << "sequence length" << " | "
              << std::setw(timesWidth) << "repeated times" << " | "
              << std::setw(reversedWidth) << "is reversed" << " |" << std::endl;
    std::cout << "|" << std::string(idWidth + 2, '-') << "|"
              << std::string(seqWidth + 2, '-') << "|"
              << std::string(posWidth + 2, '-') << "|"
              << std::string(lenWidth + 2, '-') << "|"
              << std::string(timesWidth + 2, '-') << "|"
              << std::string(reversedWidth + 2, '-') << "|" << std::endl;
    for (size_t i = 0; i < results.size(); i++) {
        std::cout << "| " << std::left << std::setw(idWidth) << i + 1 << " | "
                  << std::setw(seqWidth) << results[i].fragment << " | "
                  << std::setw(posWidth) << results[i].pos << " | "
                  << std::setw(lenWidth) << results[i].length << " | "
                  << std::setw(timesWidth) << results[i].times << " | "
                  << std::setw(reversedWidth) << (results[i].is_reversed ? "true" : "false") << " |" << std::endl;
    }
    return 0;
}