from chanfig import Config


class LossConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        
        self.distogram = Config(
            min_bin = 2.3125,
            max_bin = 21.6875,
            num_bins = 64,
            eps = 1e-6,
            weight = 0.3,
        )
        self.experimentally_resolved = Config(
            eps = 1e-8,
            min_resolution = 0.1,
            max_resolution = 3.0,
            weight = 0.0,
        )
        self.fape = Config(
            backbone = Config(
                clamp_distance = 10.0,
                clamp_distance_between_chains = 30.0,
                loss_unit_distance = 10.0,
                loss_unit_distance_between_chains = 20.0,
                weight = 0.5,
                eps = 1e-4,
            ),
            sidechain = Config(
                clamp_distance = 10.0,
                length_scale = 10.0,
                weight = 0.5,
                eps = 1e-4,
            ),
            weight = 1.0,
        )
        self.plddt = Config(
            min_resolution = 0.1,
            max_resolution = 3.0,
            cutoff = 15.0,
            num_bins = 50,
            eps = 1e-10,
            weight = 0.01,
        )
        self.masked_msa = Config(
            eps = 1e-8,
            weight = 2.0,
        )
        self.supervised_chi = Config(
            chi_weight = 0.5,
            angle_norm_weight = 0.01,
            eps = 1e-6,
            weight = 1.0,
        )
        self.violation = Config(
            violation_tolerance_factor = 12.0,
            clash_overlap_tolerance = 1.5,
            bond_angle_loss_weight = 0.3,
            eps = 1e-6,
            weight = 0.0,
        )
        self.pae = Config(
            max_bin = 31,
            num_bins = 64,
            min_resolution = 0.1,
            max_resolution = 3.0,
            eps = 1e-8,
            weight = 0.0,
        )
        self.repr_norm = Config(
            weight = 0.01,
            tolerance = 1.0,
        )
        self.chain_centre_mass = Config(
            weight = 0.0,
            eps = 1e-8,
        )
