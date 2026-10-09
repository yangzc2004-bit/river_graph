# K3 joint local--message screening pilot v2

The joint model is the same additive architecture as `k3_joint_spec.md`. This
screen uses eight maximum epochs and patience three to decide whether a longer
confirmation is worth the cost. It has six products: DOC, two masks and three
seeds. The original 20-epoch budget remains reserved for confirmation and is
not silently claimed by this screening run.
