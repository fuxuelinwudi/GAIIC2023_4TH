# coding:utf-8


def get_position_penalty_value(max_length, num_candidates=16):
    """
    length penalty

    gamma_t = 1 / (|candidates| + 1 - t) ** 2, where t = |true_label|
    gamma_t_new = |candidates| * gamma_t * v
    """

    if num_candidates == 0:
        v = 1
    else:
        v = 0
        for i in range(1, num_candidates + 1):
            v = v + (1 / i ** 2)

    gamma_t = []
    for i in range(1, max_length + 1):
        try:
            value = 1 / (num_candidates + 1 - i) ** 2
            gamma_t.append(value)
        except:
            gamma_t.append(0)

    gamma_t_new = [i * num_candidates * v for i in gamma_t]

    return gamma_t_new


def get_label_max_length(labels):
    max_n = []
    for label in labels:
        n = label[label != -100]
        max_n.append(len(n))
    return max(max_n)
