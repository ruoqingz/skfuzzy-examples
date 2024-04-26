import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def sugeno_fuzzy_inference(x_score, y_score, output_mf):
    # x is input1 and y is input2
    x = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'x')
    y = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'y')

    x['low'] = fuzz.gaussmf(x.universe, 0, 1.7)
    x['medium'] = fuzz.gaussmf(x.universe, 5, 1.7)
    x['high'] = fuzz.gaussmf(x.universe, 10, 1.7)
    y['low'] = fuzz.gaussmf(y.universe, 0, 1.7)
    y['medium'] = fuzz.gaussmf(y.universe, 5, 1.7)
    y['high'] = fuzz.gaussmf(y.universe, 10, 1.7)

    # interpret the rule
    w = []  # compose of 9 elements
    z = []  # compose of 9 elements
    count = 0
    for termx in x.terms.values():
        for termy in y.terms.values():
            if termx.label == 'low':
                count += 1
            elif termx.label == 'medium':
                count += 2
            else:
                count += 3
            if termy.label == 'low':
                count += 1
            elif termy.label == 'medium':
                count += 2
            else:
                count += 3

            if count == 2 or count == 3:
                z.append(output_mf[0][0] * x_score + output_mf[0][1] * y_score + output_mf[0][2])  # bad
            elif count == 4:
                z.append(output_mf[1][0] * x_score + output_mf[1][1] * y_score + output_mf[1][2])  # fair
            elif count == 5 or count == 6:
                z.append(output_mf[2][0] * x_score + output_mf[2][1] * y_score + output_mf[2][2])  # good

            w.append(np.fmin(fuzz.interp_membership(x.universe, x[termx.label].mf, x_score),
                             fuzz.interp_membership(y.universe, y[termy.label].mf, y_score)))

            count = 0

    numerator = np.sum(np.multiply(w, z))
    denominator = np.sum(w)
    return numerator / denominator


if __name__ == '__main__':
    # a list of inputs
    inputs = [1, 7]
    # a list of lists of several elements (every element is the coefficient of the membership function)
    output_mf = [[0, 0, 0], [0, 0, 5], [0, 0, 10]]

    # generate 9 fuzzy rules
    # if x is low and y is low then output is bad
    # if x is low and y is medium then output is bad
    # if x is low and y is high then output is fair
    # if x is medium and y is low then output is bad
    # if x is medium and y is medium then output is fair
    # if x is medium and y is high then output is good
    # if x is high and y is low then output is fair
    # if x is high and y is medium then output is good
    # if x is high and y is high then output is good

    output = sugeno_fuzzy_inference(inputs[0], inputs[1], output_mf)

    print(output)
