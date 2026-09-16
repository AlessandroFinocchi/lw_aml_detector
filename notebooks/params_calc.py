from itertools import product


def num_neurons_adv_tr(f, adv_dims):
    dims = [f] + list(adv_dims) + [1]
    return sum((dims[i-1] + 1) * dims[i] for i in range(1, len(dims)))


def num_neurons_det(f, bb_dims, det_dims, wrap_at):
    bb = [f] + list(bb_dims) + [1]
    sum_dims = sum((bb[i-1] + 1) * bb[i] for i in range(1, len(bb)))

    for layer in wrap_at:
        det = [bb_dims[layer]] + list(det_dims) + [1]
        sum_dims += sum((det[i-1] + 1) * det[i] for i in range(1, len(det)))

    return sum_dims


dataset_feats = [10, 44, 70]

adv_dims = [(48,16), (80, 32), (128, 64), (192, 64, 16), (256, 128, 32), (768, 320, 96)]

det_bb_dims_2 = [(32,16), (64, 32), (128, 64)]
det_dims_2 = [(32,16), (64,32)]
det_wrap_at_2 = [(0,), (1,), (0,1)]

det_bb_dims_3 = [(128, 64, 16), (256, 128, 32), (512, 256, 96)]
det_dims_3 = [(64,32), (128,64)]
det_wrap_at_3 = [(0,), (2,), (0,2), (0,1,2)]


def print_det_grid(name, f, bb_dims_list, det_dims_list, wrap_at_list):
    print(f"  [{name}]")
    for bb_dims, det_dims, wrap_at in product(bb_dims_list, det_dims_list, wrap_at_list):
        n = num_neurons_det(f, bb_dims=bb_dims, det_dims=det_dims, wrap_at=wrap_at)
        print(f"    bb={bb_dims}\tdet={det_dims}\twrap_at={wrap_at}\t-> {n}")


for f in dataset_feats:
    print(f"=== features: {f} ===")

    print("  [adv_tr]")
    for d in adv_dims:
        print(f"    adv={d}\t-> {num_neurons_adv_tr(f, adv_dims=d)}")

    print_det_grid("det_2", f, det_bb_dims_2, det_dims_2, det_wrap_at_2)
    print_det_grid("det_3", f, det_bb_dims_3, det_dims_3, det_wrap_at_3)

    print()
