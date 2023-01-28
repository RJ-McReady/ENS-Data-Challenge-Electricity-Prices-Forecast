def list_to_str(my_list, separator=", ", brackets="{}"):
    left, right = brackets
    if len(my_list) == 0:
        return brackets
    else:
        my_str = f"{left}{my_list[0]}"
        for l in my_list[1:]:
            my_str += separator
            my_str += str(l)
        my_str += right
        return my_str


def dict_to_str(my_dict, link="=", separator=", ", brackets="{}"):
    my_list = [f"{key}{link}{value}" for key, value in my_dict.items()]
    return list_to_str(my_list, separator=separator, brackets=brackets)


if __name__ == "__main__":
    print(list_to_str([]))
    print(list_to_str(["4, 6, 8"], brackets=("<-", "->")))
    print(list_to_str(["snake", "eater"], separator="--", brackets=("xX", "Xx")))
    print(dict_to_str({6: 9, "blinx": 2}))
