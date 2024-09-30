class NestedListIndex:
    def __init__(self, nested_list_lengths):
        self.nested_list_lengths = nested_list_lengths
        self.cumulative_lengths = self._calculate_cumulative_lengths()
        self.total_length = self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def _calculate_cumulative_lengths(self):
        cumulative = [0]
        for sublist_len in self.nested_list_lengths:
            cumulative.append(cumulative[-1] + sublist_len)
        return cumulative

    def find_list_index(self, n):
        if n < 0 or n >= self.total_length:
            raise IndexError("Index out of range")

        # 二分查找
        left, right = 0, len(self.cumulative_lengths) - 1
        while left < right:
            mid = (left + right) // 2
            if self.cumulative_lengths[mid] <= n:
                left = mid + 1
            else:
                right = mid

        list_index = left - 1
        element_index = n - self.cumulative_lengths[list_index]

        return list_index, element_index
