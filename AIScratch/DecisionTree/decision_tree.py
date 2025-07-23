import pandas as pd

class DecisionTree():
    def __init__(self, depth = 1, max_depth = -1, min_rows = -1):
        self.depth = depth
        self.max_depth = max_depth
        self.min_rows = min_rows
        self.column_split = None
        self.treshold = None
        self.decision = None
        self.ltree = None
        self.rtree = None
        self.type = None
        
    def gini_purity(self, data_frame : pd.DataFrame, sep : str):
        class_counts = data_frame[sep].value_counts()
        total = len(data_frame)
        gini = 1.0
        for count in class_counts:
            prob = count / total
            gini -= prob ** 2
        return gini

    def __split_object(self, df : pd.DataFrame, sep : str, column : str):
        class_names = df[column].value_counts().keys()
        treshold = None
        global_purity = 1 # max purity
        for _class in class_names:
            df1 = df[df[column] == _class]
            df2 = df[df[column] != _class]
            purity_1 = self.gini_purity(df1, sep)
            purity_2 = self.gini_purity(df2, sep)
            tmp_purity = purity_1 * (len(df1) / len(df)) + purity_2 * (len(df2) / len(df))
            if tmp_purity < global_purity:
                treshold = _class
                global_purity = tmp_purity
        return treshold, global_purity
    def __split_numerical(self, df : pd.DataFrame, sep : str, column : str):
        df_sorted = df.sort_values(by=column, ascending=True)
        treshold = None
        global_purity = 1 # max purity
        for i in range(len(df_sorted[column])):
            if i == len(df_sorted[column]) - 1:
                break
            tmp_treshold = (df_sorted.iloc[i][column] + df_sorted.iloc[i + 1][column]) / 2
            df1 = df[df[column] <= tmp_treshold]
            df2 = df[df[column] > tmp_treshold]
            purity_1 = self.gini_purity(df1, sep)
            purity_2 = self.gini_purity(df2, sep)
            tmp_purity = purity_1 * (len(df1) / len(df)) + purity_2 * (len(df2) / len(df))
            if tmp_purity < global_purity:
                treshold = tmp_treshold
                global_purity = tmp_purity
        return treshold, global_purity
    def find_split(self, df : pd.DataFrame, sep : str):
        column_split = None
        treshold = None
        global_purity = 1 # max purity
        for column in df.columns:
            if column == sep:
                continue
            if df.dtypes[column] == "object":
                tmp_treshold, tmp_purity = self.__split_object(df, sep, column)
            else:
                tmp_treshold, tmp_purity = self.__split_numerical(df, sep, column)
            if tmp_purity < global_purity:
                column_split = column
                treshold = tmp_treshold
                global_purity = tmp_purity
            if tmp_purity == 0:
                break
        return column_split, treshold

    def build_tree(self, df : pd.DataFrame, sep : str):
        if (self.max_depth > -1 and self.depth >= self.max_depth) \
        or (self.min_rows > -1 and len(df) <= self.min_rows): # resolve stop conditions
            final_separation = df[sep].value_counts()
            self.decision = {i : final_separation[i]/len(df) for i in final_separation.keys()}
            return # no more
        if self.gini_purity(df, sep) == 0:# completely pure
            self.decision = {i: 1.0 for i in df[sep].unique()}
            return
        self.column_split, self.treshold = self.find_split(df, sep)
        if df.dtypes[self.column_split] == "object":
            self.type = "o" # object
            df1 = df[df[self.column_split] == self.treshold]
            df2 = df[df[self.column_split] != self.treshold]
        else:
            self.type = "n" # numerical
            df1 = df[df[self.column_split] <= self.treshold]
            df2 = df[df[self.column_split] > self.treshold]
        self.ltree = DecisionTree(self.depth + 1, self.max_depth, self.min_rows)
        self.ltree.build_tree(df1, sep)
        self.rtree = DecisionTree(self.depth + 1, self.max_depth, self.min_rows)
        self.rtree.build_tree(df2, sep)

    def __predict_decision(self, sample):
        if self.decision != None:
            return self.decision
        if self.type == "o":
            return self.ltree.__predict_decision(sample) if sample[self.column_split] == self.treshold else self.rtree.__predict_decision(sample)
        return self.ltree.__predict_decision(sample) if sample[self.column_split] <= self.treshold else self.rtree.__predict_decision(sample)
    def predict(self, sample):
        decision = self.__predict_decision(sample)
        return max(decision, key=decision.get)

    def print_tree(self, indent=""):
        if self.decision is not None:
            print(indent + " Leaf:", self.decision)
        else:
            if self.type == "o":
                print(indent + f"[{self.column_split} == {self.treshold}]")
            else:
                print(indent + f"[{self.column_split} <= {self.treshold}]")
            if self.ltree:
                self.ltree.print_tree(indent + "  L-")
            if self.rtree:
                self.rtree.print_tree(indent + "  R-")

    def __write_tree(self, f, indent=""):
        if self.decision is not None:
            f.write(indent + "L:" + str(self.decision) + "\n")
        else:
            if self.type == "o":
                f.write(indent + f"[{self.column_split} == {self.treshold}]\n")
            else:
                f.write(indent + f"[{self.column_split} <= {self.treshold}]\n")
            if self.ltree:
                self.ltree.__write_tree(f, indent + "-")
            if self.rtree:
                self.rtree.__write_tree(f, indent + "+")
    def save_tree(self, file_path):
        with open(file_path, "w") as f:
            self.__write_tree(f)

    @staticmethod
    def __parse_lines(lines : list[str], depth = 1):
        line = lines.pop(0)
        indent = len(line) - len(line.lstrip("-+"))
        content = line.lstrip("-+")
        node = DecisionTree(depth)
        # leaf
        if content.startswith("L"):
            node.decision = eval(content[2:])
            return node
        # issue : not leaf but not branch
        if not "<=" in content and not "==" in content:
            return None
        # object/numerical branch
        if "==" in content:
            col, val = content.strip("[]").split(" == ")
            node.type = "o"
        else: 
            col, val = content.strip("[]").split(" <= ")
            val = float(val)
            node.type = "n"   
        node.column_split = col
        node.treshold = val
        if lines and lines[0][indent] == "-":
            node.ltree = DecisionTree.__parse_lines(lines, depth + 1)
        if lines and lines[0][indent] == "+":
            node.rtree = DecisionTree.__parse_lines(lines, depth + 1)
        return node

    @staticmethod
    def load_tree(file_path):
        with open(file_path, "r") as f:
            lines = [line.rstrip() for line in f.readlines()]
        return DecisionTree.__parse_lines(lines)