from chanfig import Variable


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