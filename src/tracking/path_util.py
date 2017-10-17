class PathUtil:
    @staticmethod
    def pad_zeros(num: int, max_digit_len: int):
        str_num = str(num)
        diff = max_digit_len - len(str_num)

        return "0" * diff + str_num
