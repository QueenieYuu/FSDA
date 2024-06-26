# Policy found on CIFAR-10 and CIFAR-100
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

from perturbations.augmentations import get_augment, augment_list, fda_choice


def arsaug_policy():
    exp0_0 = [
        [('Solarize', 0.66, 0.34), ('Equalize', 0.56, 0.61)],
        [('Equalize', 0.43, 0.06), ('AutoContrast', 0.66, 0.08)],
        [('Color', 0.72, 0.47), ('Contrast', 0.88, 0.86)],
        [('Brightness', 0.84, 0.71), ('Color', 0.31, 0.74)],
        [('Rotate', 0.68, 0.26), ('TranslateX', 0.38, 0.88)]]
    exp0_1 = [
        [('TranslateY', 0.88, 0.96), ('TranslateY', 0.53, 0.79)],
        [('AutoContrast', 0.44, 0.36), ('Solarize', 0.22, 0.48)],
        [('AutoContrast', 0.93, 0.32), ('Solarize', 0.85, 0.26)],
        [('Solarize', 0.55, 0.38), ('Equalize', 0.43, 0.48)],
        [('TranslateY', 0.72, 0.93), ('AutoContrast', 0.83, 0.95)]]
    exp0_2 = [
        [('Solarize', 0.43, 0.58), ('AutoContrast', 0.82, 0.26)],
        [('TranslateY', 0.71, 0.79), ('AutoContrast', 0.81, 0.94)],
        [('AutoContrast', 0.92, 0.18), ('TranslateY', 0.77, 0.85)],
        [('Equalize', 0.71, 0.69), ('Color', 0.23, 0.33)],
        [('Sharpness', 0.36, 0.98), ('Brightness', 0.72, 0.78)]]
    exp0_3 = [
        [('Equalize', 0.74, 0.49), ('TranslateY', 0.86, 0.91)],
        [('TranslateY', 0.82, 0.91), ('TranslateY', 0.96, 0.79)],
        [('AutoContrast', 0.53, 0.37), ('Solarize', 0.39, 0.47)],
        [('TranslateY', 0.22, 0.78), ('Color', 0.91, 0.65)],
        [('Brightness', 0.82, 0.46), ('Color', 0.23, 0.91)]]
    exp0_4 = [
        [('Cutout', 0.27, 0.45), ('Equalize', 0.37, 0.21)],
        [('Color', 0.43, 0.23), ('Brightness', 0.65, 0.71)],
        [('ShearX', 0.49, 0.31), ('AutoContrast', 0.92, 0.28)],
        [('Equalize', 0.62, 0.59), ('Equalize', 0.38, 0.91)],
        [('Solarize', 0.57, 0.31), ('Equalize', 0.61, 0.51)]]

    exp0_5 = [
        [('TranslateY', 0.29, 0.35), ('Sharpness', 0.31, 0.64)],
        [('Color', 0.73, 0.77), ('TranslateX', 0.65, 0.76)],
        [('ShearY', 0.29, 0.74), ('Posterize', 0.42, 0.58)],
        [('Color', 0.92, 0.79), ('Equalize', 0.68, 0.54)],
        [('Sharpness', 0.87, 0.91), ('Sharpness', 0.93, 0.41)]]
    exp0_6 = [
        [('Solarize', 0.39, 0.35), ('Color', 0.31, 0.44)],
        [('Color', 0.33, 0.77), ('Color', 0.25, 0.46)],
        [('ShearY', 0.29, 0.74), ('Posterize', 0.42, 0.58)],
        [('AutoContrast', 0.32, 0.79), ('Cutout', 0.68, 0.34)],
        [('AutoContrast', 0.67, 0.91), ('AutoContrast', 0.73, 0.83)]]

    return exp0_0 + exp0_1 + exp0_2 + exp0_3 + exp0_4 + exp0_5 + exp0_6


def autoaug2arsaug(f):
    def autoaug():
        mapper = defaultdict(lambda: lambda x: x)
        mapper.update({
            'ShearX': lambda x: float_parameter(x, 0.3),
            'ShearY': lambda x: float_parameter(x, 0.3),
            'TranslateX': lambda x: int_parameter(x, 10),
            'TranslateY': lambda x: int_parameter(x, 10),
            'Rotate': lambda x: int_parameter(x, 30),
            'Solarize': lambda x: 256 - int_parameter(x, 256),
            'Posterize2': lambda x: 4 - int_parameter(x, 4),
            'Contrast': lambda x: float_parameter(x, 1.8) + .1,
            'Color': lambda x: float_parameter(x, 1.8) + .1,
            'Brightness': lambda x: float_parameter(x, 1.8) + .1,
            'Sharpness': lambda x: float_parameter(x, 1.8) + .1,
            'CutoutAbs': lambda x: int_parameter(x, 20)
        })

        def low_high(name, prev_value):
            _, low, high = get_augment(name)
            return float(prev_value - low) / (high - low)

        policies = f()
        new_policies = []
        for policy in policies:
            new_policies.append([(name, pr, low_high(name, mapper[name](level))) for name, pr, level in policy])
        return new_policies

    return autoaug


@autoaug2arsaug
def autoaug_paper_cifar10():
    return [
        [('Invert', 0.1, 7), ('Contrast', 0.2, 6)],
        [('Rotate', 0.7, 2), ('TranslateXAbs', 0.3, 9)],
        [('Sharpness', 0.8, 1), ('Sharpness', 0.9, 3)],
        [('ShearY', 0.5, 8), ('TranslateYAbs', 0.7, 9)],
        [('AutoContrast', 0.5, 8), ('Equalize', 0.9, 2)],
        [('ShearY', 0.2, 7), ('Posterize2', 0.3, 7)],
        [('Color', 0.4, 3), ('Brightness', 0.6, 7)],
        [('Sharpness', 0.3, 9), ('Brightness', 0.7, 9)],
        [('Equalize', 0.6, 5), ('Equalize', 0.5, 1)],
        [('Contrast', 0.6, 7), ('Sharpness', 0.6, 5)],
        [('Color', 0.7, 7), ('TranslateXAbs', 0.5, 8)],
        [('Equalize', 0.3, 7), ('AutoContrast', 0.4, 8)],
        [('TranslateYAbs', 0.4, 3), ('Sharpness', 0.2, 6)],
        [('Brightness', 0.9, 6), ('Color', 0.2, 6)],
        [('Solarize', 0.5, 2), ('Invert', 0.0, 3)],
        [('Equalize', 0.2, 0), ('AutoContrast', 0.6, 0)],
        [('Equalize', 0.2, 8), ('Equalize', 0.6, 4)],
        [('Color', 0.9, 9), ('Equalize', 0.6, 6)],
        [('AutoContrast', 0.8, 4), ('Solarize', 0.2, 8)],
        [('Brightness', 0.1, 3), ('Color', 0.7, 0)],
        [('Solarize', 0.4, 5), ('AutoContrast', 0.9, 3)],
        [('TranslateYAbs', 0.9, 9), ('TranslateYAbs', 0.7, 9)],
        [('AutoContrast', 0.9, 2), ('Solarize', 0.8, 3)],
        [('Equalize', 0.8, 8), ('Invert', 0.1, 3)],
        [('TranslateYAbs', 0.7, 9), ('AutoContrast', 0.9, 1)],
    ]


@autoaug2arsaug
def autoaug_policy():
    """AutoAugment policies found on Cifar."""
    exp0_0 = [
        [('Invert', 0.1, 7), ('Contrast', 0.2, 6)],
        [('Rotate', 0.7, 2), ('TranslateXAbs', 0.3, 9)],
        [('Sharpness', 0.8, 1), ('Sharpness', 0.9, 3)],
        [('ShearY', 0.5, 8), ('TranslateYAbs', 0.7, 9)],
        [('AutoContrast', 0.5, 8), ('Equalize', 0.9, 2)]]
    exp0_1 = [
        [('Solarize', 0.4, 5), ('AutoContrast', 0.9, 3)],
        [('TranslateYAbs', 0.9, 9), ('TranslateYAbs', 0.7, 9)],
        [('AutoContrast', 0.9, 2), ('Solarize', 0.8, 3)],
        [('Equalize', 0.8, 8), ('Invert', 0.1, 3)],
        [('TranslateYAbs', 0.7, 9), ('AutoContrast', 0.9, 1)]]
    exp0_2 = [
        [('Solarize', 0.4, 5), ('AutoContrast', 0.0, 2)],
        [('TranslateYAbs', 0.7, 9), ('TranslateYAbs', 0.7, 9)],
        [('AutoContrast', 0.9, 0), ('Solarize', 0.4, 3)],
        [('Equalize', 0.7, 5), ('Invert', 0.1, 3)],
        [('TranslateYAbs', 0.7, 9), ('TranslateYAbs', 0.7, 9)]]
    exp0_3 = [
        [('Solarize', 0.4, 5), ('AutoContrast', 0.9, 1)],
        [('TranslateYAbs', 0.8, 9), ('TranslateYAbs', 0.9, 9)],
        [('AutoContrast', 0.8, 0), ('TranslateYAbs', 0.7, 9)],
        [('TranslateYAbs', 0.2, 7), ('Color', 0.9, 6)],
        [('Equalize', 0.7, 6), ('Color', 0.4, 9)]]
    exp1_0 = [
        [('ShearY', 0.2, 7), ('Posterize2', 0.3, 7)],
        [('Color', 0.4, 3), ('Brightness', 0.6, 7)],
        [('Sharpness', 0.3, 9), ('Brightness', 0.7, 9)],
        [('Equalize', 0.6, 5), ('Equalize', 0.5, 1)],
        [('Contrast', 0.6, 7), ('Sharpness', 0.6, 5)]]
    exp1_1 = [
        [('Brightness', 0.3, 7), ('AutoContrast', 0.5, 8)],
        [('AutoContrast', 0.9, 4), ('AutoContrast', 0.5, 6)],
        [('Solarize', 0.3, 5), ('Equalize', 0.6, 5)],
        [('TranslateYAbs', 0.2, 4), ('Sharpness', 0.3, 3)],
        [('Brightness', 0.0, 8), ('Color', 0.8, 8)]]
    exp1_2 = [
        [('Solarize', 0.2, 6), ('Color', 0.8, 6)],
        [('Solarize', 0.2, 6), ('AutoContrast', 0.8, 1)],
        [('Solarize', 0.4, 1), ('Equalize', 0.6, 5)],
        [('Brightness', 0.0, 0), ('Solarize', 0.5, 2)],
        [('AutoContrast', 0.9, 5), ('Brightness', 0.5, 3)]]
    exp1_3 = [
        [('Contrast', 0.7, 5), ('Brightness', 0.0, 2)],
        [('Solarize', 0.2, 8), ('Solarize', 0.1, 5)],
        [('Contrast', 0.5, 1), ('TranslateYAbs', 0.2, 9)],
        [('AutoContrast', 0.6, 5), ('TranslateYAbs', 0.0, 9)],
        [('AutoContrast', 0.9, 4), ('Equalize', 0.8, 4)]]
    exp1_4 = [
        [('Brightness', 0.0, 7), ('Equalize', 0.4, 7)],
        [('Solarize', 0.2, 5), ('Equalize', 0.7, 5)],
        [('Equalize', 0.6, 8), ('Color', 0.6, 2)],
        [('Color', 0.3, 7), ('Color', 0.2, 4)],
        [('AutoContrast', 0.5, 2), ('Solarize', 0.7, 2)]]
    exp1_5 = [
        [('AutoContrast', 0.2, 0), ('Equalize', 0.1, 0)],
        [('ShearY', 0.6, 5), ('Equalize', 0.6, 5)],
        [('Brightness', 0.9, 3), ('AutoContrast', 0.4, 1)],
        [('Equalize', 0.8, 8), ('Equalize', 0.7, 7)],
        [('Equalize', 0.7, 7), ('Solarize', 0.5, 0)]]
    exp1_6 = [
        [('Equalize', 0.8, 4), ('TranslateYAbs', 0.8, 9)],
        [('TranslateYAbs', 0.8, 9), ('TranslateYAbs', 0.6, 9)],
        [('TranslateYAbs', 0.9, 0), ('TranslateYAbs', 0.5, 9)],
        [('AutoContrast', 0.5, 3), ('Solarize', 0.3, 4)],
        [('Solarize', 0.5, 3), ('Equalize', 0.4, 4)]]
    exp2_0 = [
        [('Color', 0.7, 7), ('TranslateXAbs', 0.5, 8)],
        [('Equalize', 0.3, 7), ('AutoContrast', 0.4, 8)],
        [('TranslateYAbs', 0.4, 3), ('Sharpness', 0.2, 6)],
        [('Brightness', 0.9, 6), ('Color', 0.2, 8)],
        [('Solarize', 0.5, 2), ('Invert', 0.0, 3)]]
    exp2_1 = [
        [('AutoContrast', 0.1, 5), ('Brightness', 0.0, 0)],
        [('CutoutAbs', 0.2, 4), ('Equalize', 0.1, 1)],
        [('Equalize', 0.7, 7), ('AutoContrast', 0.6, 4)],
        [('Color', 0.1, 8), ('ShearY', 0.2, 3)],
        [('ShearY', 0.4, 2), ('Rotate', 0.7, 0)]]
    exp2_2 = [
        [('ShearY', 0.1, 3), ('AutoContrast', 0.9, 5)],
        [('TranslateYAbs', 0.3, 6), ('CutoutAbs', 0.3, 3)],
        [('Equalize', 0.5, 0), ('Solarize', 0.6, 6)],
        [('AutoContrast', 0.3, 5), ('Rotate', 0.2, 7)],
        [('Equalize', 0.8, 2), ('Invert', 0.4, 0)]]
    exp2_3 = [
        [('Equalize', 0.9, 5), ('Color', 0.7, 0)],
        [('Equalize', 0.1, 1), ('ShearY', 0.1, 3)],
        [('AutoContrast', 0.7, 3), ('Equalize', 0.7, 0)],
        [('Brightness', 0.5, 1), ('Contrast', 0.1, 7)],
        [('Contrast', 0.1, 4), ('Solarize', 0.6, 5)]]
    exp2_4 = [
        [('Solarize', 0.2, 3), ('ShearX', 0.0, 0)],
        [('TranslateXAbs', 0.3, 0), ('TranslateXAbs', 0.6, 0)],
        [('Equalize', 0.5, 9), ('TranslateYAbs', 0.6, 7)],
        [('ShearX', 0.1, 0), ('Sharpness', 0.5, 1)],
        [('Equalize', 0.8, 6), ('Invert', 0.3, 6)]]
    exp2_5 = [
        [('AutoContrast', 0.3, 9), ('CutoutAbs', 0.5, 3)],
        [('ShearX', 0.4, 4), ('AutoContrast', 0.9, 2)],
        [('ShearX', 0.0, 3), ('Posterize2', 0.0, 3)],
        [('Solarize', 0.4, 3), ('Color', 0.2, 4)],
        [('Equalize', 0.1, 4), ('Equalize', 0.7, 6)]]
    exp2_6 = [
        [('Equalize', 0.3, 8), ('AutoContrast', 0.4, 3)],
        [('Solarize', 0.6, 4), ('AutoContrast', 0.7, 6)],
        [('AutoContrast', 0.2, 9), ('Brightness', 0.4, 8)],
        [('Equalize', 0.1, 0), ('Equalize', 0.0, 6)],
        [('Equalize', 0.8, 4), ('Equalize', 0.0, 4)]]
    exp2_7 = [
        [('Equalize', 0.5, 5), ('AutoContrast', 0.1, 2)],
        [('Solarize', 0.5, 5), ('AutoContrast', 0.9, 5)],
        [('AutoContrast', 0.6, 1), ('AutoContrast', 0.7, 8)],
        [('Equalize', 0.2, 0), ('AutoContrast', 0.1, 2)],
        [('Equalize', 0.6, 9), ('Equalize', 0.4, 4)]]
    exp0s = exp0_0 + exp0_1 + exp0_2 + exp0_3
    exp1s = exp1_0 + exp1_1 + exp1_2 + exp1_3 + exp1_4 + exp1_5 + exp1_6
    exp2s = exp2_0 + exp2_1 + exp2_2 + exp2_3 + exp2_4 + exp2_5 + exp2_6 + exp2_7

    return exp0s + exp1s + exp2s


PARAMETER_MAX = 10


def float_parameter(level, maxval):
    return float(level) * maxval / PARAMETER_MAX


def int_parameter(level, maxval):
    return int(float_parameter(level, maxval))


def no_duplicates(f):
    def wrap_remove_duplicates():
        policies = f()
        return remove_deplicates(policies)

    return wrap_remove_duplicates


def remove_deplicates(policies):
    s = set()
    new_policies = []
    for ops in policies:
        key = []
        for op in ops:
            key.append(op[0])
        key = '_'.join(key)
        if key in s:
            continue
        else:
            s.add(key)
            new_policies.append(ops)

    return new_policies

def policy_decoder(augment, num_policy, num_op):
    op_list = fda_choice(False)
    policies = []
    for i in range(num_policy):
        ops = []
        for j in range(num_op):
            op_idx = augment['policy_%d_%d' % (i, j)]
            op_prob = augment['prob_%d_%d' % (i, j)]
            op_level = augment['level_%d_%d' % (i, j)]
            ops.append((op_list[op_idx][0].__name__, op_prob, op_level))
        policies.append(ops)
    return policies
