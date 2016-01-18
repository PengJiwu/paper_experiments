__author__ = 'tao'
class Solution(object):
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        ix = self.find_small_ix(nums)
        if ix == -1:
            nums[:] = sorted(nums)
        else:
            tmp = self.find_large_than_num_ix(nums[ix], nums[ix+1:])
            change_ix = len(nums) - tmp
            num = nums[ix]
            nums[ix] = nums[change_ix]
            nums[change_ix] = num

            nums[ix+1:] = sorted(nums[ix+1:])


    def find_large_than_num_ix(self, num, sublist):
        max = -1
        for ix in range(len(sublist)):
            if sublist[ix] > num:
                if max == -1 or sublist[ix] < sublist[max]:
                    max = ix
        return len(sublist) - max

    def find_small_ix(self, sublist):
        for ix in sorted(range(len(sublist)-1), reverse=True):
            if not self._is_bigger_than_all(sublist[ix], sublist[ix+1:]):
                return ix
        return -1

    def _is_bigger_than_all(self, num, sublist):
        result = True
        for element in sublist:
            if num < element:
                result = False
                break
        return result

s = Solution()
a = [3, 2, 1]
s.nextPermutation(a)
print a
