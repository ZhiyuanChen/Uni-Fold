from chanfig import Config, Variable

from .data import SupervisedData, TrainData, EvalData, PredictData

N_RES = "number of residues"
N_MSA = "number of MSA sequences"
N_EXTRA_MSA = "number of extra MSA sequences"
N_TPL = "number of templates"


d_pair = Variable(128)
d_msa = Variable(256)
d_template = Variable(64)
d_extra_msa = Variable(64)
d_single = Variable(384)
max_recycling_iters = Variable(3)
chunk_size = Variable(4)
aux_distogram_bins = Variable(64)
eps = Variable(1e-8)
inf = Variable(3e4)
use_templates = Variable(True)
is_multimer = Variable(False)


class DataConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.common = CommonData()
        self.supervised = SupervisedData()
        self.train = TrainData()
        self.eval = EvalData()
        self.predict = PredictData()


class CommonData(Config):
    def __init__(self) -> None:
        super().__init__()
        self.features.aatype = [N_RES]
        self.features.all_atom_mask = [N_RES, None]
        self.features.all_atom_positions = [N_RES, None, None]
        self.features.alt_chi_angles = [N_RES, None]
        self.features.atom14_alt_gt_exists = [N_RES, None]
        self.features.atom14_alt_gt_positions = [N_RES, None, None]
        self.features.atom14_atom_exists = [N_RES, None]
        self.features.atom14_atom_is_ambiguous = [N_RES, None]
        self.features.atom14_gt_exists = [N_RES, None]
        self.features.atom14_gt_positions = [N_RES, None, None]
        self.features.atom37_atom_exists = [N_RES, None]
        self.features.frame_mask = [N_RES]
        self.features.true_frame_tensor = [N_RES, None, None]
        self.features.bert_mask = [N_MSA, N_RES]
        self.features.chi_angles_sin_cos = [N_RES, None, None]
        self.features.chi_mask = [N_RES, None]
        self.features.extra_msa_deletion_value = [N_EXTRA_MSA, N_RES]
        self.features.extra_msa_has_deletion = [N_EXTRA_MSA, N_RES]
        self.features.extra_msa = [N_EXTRA_MSA, N_RES]
        self.features.extra_msa_mask = [N_EXTRA_MSA, N_RES]
        self.features.extra_msa_row_mask = [N_EXTRA_MSA]
        self.features.is_distillation = []
        self.features.msa_feat = [N_MSA, N_RES, None]
        self.features.msa_mask = [N_MSA, N_RES]
        self.features.msa_chains = [N_MSA, None]
        self.features.msa_row_mask = [N_MSA]
        self.features.num_recycling_iters = []
        self.features.pseudo_beta = [N_RES, None]
        self.features.pseudo_beta_mask = [N_RES]
        self.features.residue_index = [N_RES]
        self.features.residx_atom14_to_atom37 = [N_RES, None]
        self.features.residx_atom37_to_atom14 = [N_RES, None]
        self.features.resolution = []
        self.features.rigidgroups_alt_gt_frames = [N_RES, None, None, None]
        self.features.rigidgroups_group_exists = [N_RES, None]
        self.features.rigidgroups_group_is_ambiguous = [N_RES, None]
        self.features.rigidgroups_gt_exists = [N_RES, None]
        self.features.rigidgroups_gt_frames = [N_RES, None, None, None]
        self.features.seq_length = []
        self.features.seq_mask = [N_RES]
        self.features.target_feat = [N_RES, None]
        self.features.template_aatype = [N_TPL, N_RES]
        self.features.template_all_atom_mask = [N_TPL, N_RES, None]
        self.features.template_all_atom_positions = [N_TPL, N_RES, None, None]
        self.features.template_alt_torsion_angles_sin_cos = [
            N_TPL,
            N_RES,
            None,
            None,
        ]
        self.features.template_frame_mask = [N_TPL, N_RES]
        self.features.template_frame_tensor = [N_TPL, N_RES, None, None]
        self.features.template_mask = [N_TPL]
        self.features.template_pseudo_beta = [N_TPL, N_RES, None]
        self.features.template_pseudo_beta_mask = [N_TPL, N_RES]
        self.features.template_sum_probs = [N_TPL, None]
        self.features.template_torsion_angles_mask = [N_TPL, N_RES, None]
        self.features.template_torsion_angles_sin_cos = [N_TPL, N_RES, None, None]
        self.features.true_msa = [N_MSA, N_RES]
        self.features.use_clamped_fape = []
        self.features.assembly_num_chains = [1]
        self.features.asym_id = [N_RES]
        self.features.sym_id = [N_RES]
        self.features.entity_id = [N_RES]
        self.features.num_sym = [N_RES]
        self.features.asym_len = [None]
        self.features.cluster_bias_mask = [N_MSA]

        self.masked_msa.profile_prob = 0.1
        self.masked_msa.same_prob = 0.1
        self.masked_msa.uniform_prob = 0.1

        self.block_delete_msa.msa_fraction_per_block = 0.3
        self.block_delete_msa.randomize_num_blocks = False
        self.block_delete_msa.num_blocks = 5
        self.block_delete_msa.min_num_msa = 16

        self.random_delete_msa.max_msa_entry = 1 << 25  # := 33554432

        self.v2_feature = False
        self.gumbel_sample = False
        self.max_extra_msa = 1024
        self.msa_cluster_features = True
        self.reduce_msa_clusters_by_max_templates = True
        self.resample_msa_in_recycling = True
        self.template_features = [
            "template_all_atom_positions",
            "template_sum_probs",
            "template_aatype",
            "template_all_atom_mask",
        ]
        self.unsupervised_features = [
            "aatype",
            "residue_index",
            "msa",
            "msa_chains",
            "num_alignments",
            "seq_length",
            "between_segment_residues",
            "deletion_matrix",
            "num_recycling_iters",
            "crop_and_fix_size_seed",
        ]
        self.recycling_features = [
            "msa_chains",
            "msa_mask",
            "msa_row_mask",
            "bert_mask",
            "true_msa",
            "msa_feat",
            "extra_msa_deletion_value",
            "extra_msa_has_deletion",
            "extra_msa",
            "extra_msa_mask",
            "extra_msa_row_mask",
            "is_distillation",
        ]
        self.multimer_features = [
            "assembly_num_chains",
            "asym_id",
            "sym_id",
            "num_sym",
            "entity_id",
            "asym_len",
            "cluster_bias_mask",
        ]
        self.use_templates = use_templates
        self.is_multimer = is_multimer
        self.use_template_torsion_angles = use_templates
        self.max_recycling_iters = max_recycling_iters
