import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


class Node:
    def __init__(
        self,
        data=None,
        children=None,
        split_on=None,
        is_leaf=False,
        pred_class=None,
        is_numerical=False,
        threshold=None,
    ):
        self.data = data
        self.children = children
        self.split_on = split_on
        # Leaf node attributes
        self.is_leaf = is_leaf
        self.pred_class = pred_class
        # Numerical split attributes
        self.is_numerical = is_numerical
        self.threshold = threshold


class DecisionTreeClassifier:
    def __init__(self, feature_types=None):
        self.root = Node()
        self.feature_types = feature_types

    def fit(self, X, Y):
        self.root.data = np.column_stack([X, Y])
        self.find_best_split(self.root)

    def predict(self, X):
        predictions = np.array([self.traverse_tree(x, self.root) for x in X])
        return predictions

    @staticmethod
    def get_y(data):
        y = data[:, -1]
        return y

    @staticmethod
    def calculate_gini_impurity(Y):
        if len(Y) == 0:
            return 0

        _, labels_counts = np.unique(Y, return_counts=True)

        total_instances = len(Y)
        gini_impurity = 1 - sum(
            [(label_count / total_instances) ** 2 for label_count in labels_counts]
        )
        return gini_impurity

    @staticmethod
    def get_pred_class(Y):
        labels, labels_counts = np.unique(Y, return_counts=True)
        index = np.argmax(labels_counts)
        return labels[index]

    def find_best_split(self, node):
        # Check if node has no data or meets stopping criteria
        if node.data is None or len(node.data) == 0:
            node.is_leaf = True
            node.pred_class = None
            return

        y = self.get_y(node.data)

        # Check if the node meets the criteria to stop splitting
        if self.calculate_gini_impurity(y) == 0:
            node.is_leaf = True
            node.pred_class = self.get_pred_class(y)
            return

        # Check if we have no features left to split on
        if node.data.shape[1] <= 1:
            node.is_leaf = True
            node.pred_class = self.get_pred_class(y)
            return

        # Initialize variables for tracking the best split
        index_feature_split = -1
        min_gini_impurity = 1
        child_nodes = None
        best_threshold = None
        is_numerical = False

        # iterate over all features, ignore (y)
        for i in range(node.data.shape[1] - 1):
            if self.feature_types[i] == "numerical":
                split_result = self.split_on_numerical_feature(node.data, i)
                if split_result is not None:
                    split_nodes, weighted_impurity, threshold = split_result
                    if len(split_nodes) > 1 and weighted_impurity < min_gini_impurity:
                        child_nodes, min_gini_impurity = split_nodes, weighted_impurity
                        index_feature_split = i
                        best_threshold = threshold
                        is_numerical = True
            else:
                split_nodes, weighted_impurity = self.split_on_categorical_feature(node.data, i)
                # Only consider splits that actually partition the data
                if len(split_nodes) > 1 and weighted_impurity < min_gini_impurity:
                    child_nodes, min_gini_impurity = split_nodes, weighted_impurity
                    index_feature_split = i
                    best_threshold = None
                    is_numerical = False

        # If no valid split found, make it a leaf
        if child_nodes is None or index_feature_split == -1 or len(child_nodes) <= 1:
            node.is_leaf = True
            y = self.get_y(node.data)
            node.pred_class = self.get_pred_class(y)
            return

        node.children = child_nodes
        node.split_on = index_feature_split
        node.threshold = best_threshold
        node.is_numerical = is_numerical

        # Recursively call the best_split function for each child node
        for child_node in child_nodes.values():
            self.find_best_split(child_node)

    def split_on_categorical_feature(self, data, feat_index):
        feature_values = data[:, feat_index]
        unique_values = np.unique(feature_values)

        split_nodes = {}
        weighted_impurity = 0
        total_instances = len(data)

        for unique_value in unique_values:
            partition = data[data[:, feat_index] == unique_value, :]
            node = Node(data=partition)
            split_nodes[unique_value] = node
            partition_y = self.get_y(partition)
            node_impurity = self.calculate_gini_impurity(partition_y)
            weighted_impurity += (len(partition) / total_instances) * node_impurity

        return split_nodes, weighted_impurity

    def split_on_numerical_feature(self, data, feat_index):
        """
        Find the best threshold for splitting a numerical feature.

        Parameters:
        -----------
        data: numpy.ndarray
            The dataset to split
        feat_index: int
            Index of the numerical feature

        Returns:
        -----------
        tuple: (split_nodes, weighted_impurity, threshold) or None if no valid split
        """
        feature_values = data[:, feat_index].astype(float)

        # Get unique sorted values
        unique_values = np.unique(feature_values)

        if len(unique_values) <= 1:
            return None

        # Consider thresholds as midpoints between consecutive unique values
        thresholds = [
            (unique_values[i] + unique_values[i + 1]) / 2
            for i in range(len(unique_values) - 1)
        ]

        best_impurity = 1
        best_threshold = None
        best_split = None

        total_instances = len(data)

        for threshold in thresholds:
            # Split data into left (<=) and right (>)
            left_data = data[feature_values <= threshold]
            right_data = data[feature_values > threshold]

            # Skip if split doesn't partition the data
            if len(left_data) == 0 or len(right_data) == 0:
                continue

            # Calculate weighted impurity
            left_y = self.get_y(left_data)
            right_y = self.get_y(right_data)

            left_impurity = self.calculate_gini_impurity(left_y)
            right_impurity = self.calculate_gini_impurity(right_y)

            weighted_impurity = (len(left_data) / total_instances) * left_impurity + (
                len(right_data) / total_instances
            ) * right_impurity

            if weighted_impurity < best_impurity:
                best_impurity = weighted_impurity
                best_threshold = threshold
                best_split = {
                    "left": Node(data=left_data),
                    "right": Node(data=right_data),
                }

        if best_split is None:
            return None

        return best_split, best_impurity, best_threshold

    def traverse_tree(self, x, node):
        if node.is_leaf:
            return node.pred_class

        feat_value = x[node.split_on]

        # Handle numerical splits
        if node.is_numerical:
            if self._is_numerical_value(feat_value):
                if float(feat_value) <= node.threshold:
                    key = "left"
                else:
                    key = "right"
            else:
                key = feat_value
        else:
            key = feat_value

        if key not in node.children:
            # If key not found, return most common class in this node
            y = self.get_y(node.data)
            return self.get_pred_class(y)

        predicted_class = self.traverse_tree(x, node.children[key])
        return predicted_class

    def _is_numerical_value(self, value):
        """Check if a value can be converted to float."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def plot_tree(
        self, feature_names=None, class_names=None, figsize=(16, 10), save_path=None
    ):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Calculate tree layout
        def get_tree_depth(node, depth=0):
            if node.is_leaf:
                return depth
            if not node.children:
                return depth
            return max(
                get_tree_depth(child, depth + 1) for child in node.children.values()
            )

        def count_leaves(node):
            if node.is_leaf:
                return 1
            if not node.children:
                return 1
            return sum(count_leaves(child) for child in node.children.values())

        max_depth = get_tree_depth(self.root)
        total_leaves = count_leaves(self.root)

        # Node positioning
        positions = {}
        leaf_counter = [0]

        def calculate_positions(node, depth=0, parent_x=None):
            y = 0.95 - (depth / (max_depth + 1)) * 0.85

            if node.is_leaf:
                x = (leaf_counter[0] + 0.5) / total_leaves
                leaf_counter[0] += 1
            else:
                # Calculate children positions first
                child_positions = []
                for child in node.children.values():
                    child_pos = calculate_positions(child, depth + 1, parent_x)
                    child_positions.append(child_pos[0])

                # Position at the center of children
                x = sum(child_positions) / len(child_positions)

            positions[id(node)] = (x, y)
            return (x, y)

        calculate_positions(self.root)

        # Drawing function
        def draw_node(node, depth=0):
            x, y = positions[id(node)]

            if node.is_leaf:
                # Leaf node - green box
                y_data = self.get_y(node.data)
                unique, counts = np.unique(y_data, return_counts=True)
                total = len(y_data)

                # Determine prediction label
                if class_names and node.pred_class in class_names:
                    pred_label = class_names[node.pred_class]
                else:
                    pred_label = str(node.pred_class)

                # Create distribution text
                dist_parts = []
                for u, c in zip(unique, counts):
                    class_label = class_names.get(u, str(u)) if class_names else str(u)
                    dist_parts.append(f"{class_label}: {c}")
                dist_text = "\n".join(dist_parts)

                impurity = self.calculate_gini_impurity(y_data)

                box_text = f"PREDICT: {pred_label}\n{dist_text}\nSamples: {total}\nImpurity: {impurity:.3f}"

                bbox = FancyBboxPatch(
                    (x - 0.08, y - 0.04),
                    0.16,
                    0.08,
                    boxstyle="round,pad=0.005",
                    edgecolor="darkgreen" if pred_label == "1" else "darkred",
                    facecolor="lightgreen" if pred_label == "1" else "lightcoral",
                    linewidth=2.5,
                )
                ax.add_patch(bbox)
                ax.text(
                    x,
                    y,
                    box_text,
                    ha="center",
                    va="center",
                    fontsize=9,
                    weight="bold",
                    family="monospace",
                )

            else:
                # Decision node - blue box
                if feature_names and node.split_on < len(feature_names):
                    feat_name = feature_names[node.split_on]
                else:
                    feat_name = f"Feature {node.split_on}"

                y_data = self.get_y(node.data)
                unique, counts = np.unique(y_data, return_counts=True)
                total = len(y_data)
                impurity = self.calculate_gini_impurity(y_data)

                dist_parts = []
                for u, c in zip(unique, counts):
                    class_label = class_names.get(u, str(u)) if class_names else str(u)
                    dist_parts.append(f"{class_label}: {c}")
                dist_text = " | ".join(dist_parts)

                # Add threshold information for numerical splits
                if node.is_numerical and node.threshold is not None:
                    split_info = f"{feat_name} <= {node.threshold:.3f}"
                else:
                    split_info = feat_name

                box_text = f"{split_info}\n{dist_text}\nSamples: {total}\nImpurity: {impurity:.3f}"

                bbox = FancyBboxPatch(
                    (x - 0.08, y - 0.04),
                    0.16,
                    0.08,
                    boxstyle="round,pad=0.005",
                    edgecolor="darkblue",
                    facecolor="lightblue",
                    linewidth=2.5,
                )
                ax.add_patch(bbox)
                ax.text(
                    x,
                    y,
                    box_text,
                    ha="center",
                    va="center",
                    fontsize=9,
                    weight="bold",
                    family="monospace",
                )

                # Draw edges to children
                if node.children:
                    for feat_value, child in node.children.items():
                        child_x, child_y = positions[id(child)]

                        # Draw arrow
                        arrow = FancyArrowPatch(
                            (x, y - 0.04),
                            (child_x, child_y + 0.04),
                            arrowstyle="->",
                            connectionstyle="arc3,rad=0.1",
                            linewidth=2,
                            color="gray",
                            alpha=0.7,
                        )
                        ax.add_patch(arrow)

                        # Add edge label
                        mid_x = (x + child_x) / 2
                        mid_y = (y - 0.04 + child_y + 0.04) / 2

                        # Format edge label based on split type
                        if node.is_numerical:
                            if feat_value == "left":
                                label = "Yes"
                            else:
                                label = "No"
                        else:
                            label = f"= {feat_value}"

                        ax.text(
                            mid_x,
                            mid_y,
                            label,
                            ha="center",
                            va="center",
                            fontsize=8,
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="yellow",
                                edgecolor="orange",
                                alpha=0.8,
                            ),
                            weight="bold",
                        )

                        # Recursively draw children
                        draw_node(child, depth + 1)

        # Draw the tree
        draw_node(self.root)

        # Add title and legend
        plt.title("Decision Tree Structure", fontsize=16, weight="bold", pad=40)

        # Create legend
        decision_patch = mpatches.Patch(color="lightblue", label="Decision Node")
        leaf_patch = mpatches.Patch(color="lightgreen", label="Leaf Node (Prediction)")
        plt.legend(handles=[decision_patch, leaf_patch], loc="upper left", fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ Tree diagram saved to: {save_path}")
        else:
            plt.show()
