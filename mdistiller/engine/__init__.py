from .trainer import BaseTrainer, CRDTrainer, DOT, CRDDOT, MaKDTrainer
trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "dot": DOT,
    "crd_dot": CRDDOT,
    "makd":MaKDTrainer
}
