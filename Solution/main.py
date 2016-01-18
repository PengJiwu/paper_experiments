class Solution(object):
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        if n <= 1:
            return '1'
        elif n == 2:
            return '11'

        n_str = '11'
        for ix in xrange(3, n + 1):
            nums = []
            char = n_str[0]
            count = 1
            n_str = n_str[1:]
            while n_str:
                if n_str[0] == char:
                    count += 1
                    n_str = n_str[1:]
                else:
                    nums.append(str(count))
                    nums.append(char)
                    char = n_str[0]
                    count = 1
                    n_str = n_str[1:]
            nums.append(str(count))
            nums.append(char)
            n_str = ''.join(nums)
        return n_str


if __name__ == '__main__':
    s = Solution()
    print s.countAndSay(5)