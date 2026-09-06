//! Fixed-step low-storage Runge--Kutta methods.
//!
//! Every built-in method owns one compile-time-validated JSON resource and
//! materializes only its selected recurrence. The public algorithm values are
//! zero-sized; [`ResourceLowStorageRungeKutta`] is the shared downstream
//! execution surface.

mod kernels;

pub use kernels::ResourceLowStorageRungeKutta;

use crate::tableau::define_low_storage_rk_from_file;

define_low_storage_rk_from_file!(pub Ork256, "src/tableau/resources/low_storage/ork256.json", crate = crate);
define_low_storage_rk_from_file!(pub CarpenterKennedy2N54, "src/tableau/resources/low_storage/carpenterkennedy2n54.json", crate = crate);
define_low_storage_rk_from_file!(pub Shlddrk64, "src/tableau/resources/low_storage/shlddrk64.json", crate = crate);
define_low_storage_rk_from_file!(pub Dglddrk73C, "src/tableau/resources/low_storage/dglddrk73c.json", crate = crate);
define_low_storage_rk_from_file!(pub Dglddrk84C, "src/tableau/resources/low_storage/dglddrk84c.json", crate = crate);
define_low_storage_rk_from_file!(pub Dglddrk84F, "src/tableau/resources/low_storage/dglddrk84f.json", crate = crate);
define_low_storage_rk_from_file!(pub Ndblsrk124, "src/tableau/resources/low_storage/ndblsrk124.json", crate = crate);
define_low_storage_rk_from_file!(pub Ndblsrk134, "src/tableau/resources/low_storage/ndblsrk134.json", crate = crate);
define_low_storage_rk_from_file!(pub Ndblsrk144, "src/tableau/resources/low_storage/ndblsrk144.json", crate = crate);
define_low_storage_rk_from_file!(pub RK46NL, "src/tableau/resources/low_storage/rk46nl.json", crate = crate);
define_low_storage_rk_from_file!(pub SHLDDRK52, "src/tableau/resources/low_storage/shlddrk52.json", crate = crate);

define_low_storage_rk_from_file!(pub CFRLDDRK64, "src/tableau/resources/low_storage/cfrlddrk64.json", crate = crate);
define_low_storage_rk_from_file!(pub TSLDDRK74, "src/tableau/resources/low_storage/tslddrk74.json", crate = crate);
define_low_storage_rk_from_file!(pub SHLDDRK_2N, "src/tableau/resources/low_storage/shlddrk-2n.json", crate = crate);

define_low_storage_rk_from_file!(pub ParsaniKetchesonDeconinck3S32, "src/tableau/resources/low_storage/parsaniketchesondeconinck3s32.json", crate = crate);
define_low_storage_rk_from_file!(pub ParsaniKetchesonDeconinck3S53, "src/tableau/resources/low_storage/parsaniketchesondeconinck3s53.json", crate = crate);
define_low_storage_rk_from_file!(pub ParsaniKetchesonDeconinck3S82, "src/tableau/resources/low_storage/parsaniketchesondeconinck3s82.json", crate = crate);
define_low_storage_rk_from_file!(pub ParsaniKetchesonDeconinck3S173, "src/tableau/resources/low_storage/parsaniketchesondeconinck3s173.json", crate = crate);
define_low_storage_rk_from_file!(pub ParsaniKetchesonDeconinck3S184, "src/tableau/resources/low_storage/parsaniketchesondeconinck3s184.json", crate = crate);
define_low_storage_rk_from_file!(pub ParsaniKetchesonDeconinck3S94, "src/tableau/resources/low_storage/parsaniketchesondeconinck3s94.json", crate = crate);
define_low_storage_rk_from_file!(pub ParsaniKetchesonDeconinck3S105, "src/tableau/resources/low_storage/parsaniketchesondeconinck3s105.json", crate = crate);
define_low_storage_rk_from_file!(pub ParsaniKetchesonDeconinck3S205, "src/tableau/resources/low_storage/parsaniketchesondeconinck3s205.json", crate = crate);

define_low_storage_rk_from_file!(pub RDPK3Sp35, "src/tableau/resources/low_storage/rdpk3sp35.json", crate = crate);
define_low_storage_rk_from_file!(pub RDPK3Sp49, "src/tableau/resources/low_storage/rdpk3sp49.json", crate = crate);
define_low_storage_rk_from_file!(pub RDPK3Sp510, "src/tableau/resources/low_storage/rdpk3sp510.json", crate = crate);
define_low_storage_rk_from_file!(pub RDPK3SpFSAL35, "src/tableau/resources/low_storage/rdpk3spfsal35.json", crate = crate);
define_low_storage_rk_from_file!(pub RDPK3SpFSAL49, "src/tableau/resources/low_storage/rdpk3spfsal49.json", crate = crate);
define_low_storage_rk_from_file!(pub RDPK3SpFSAL510, "src/tableau/resources/low_storage/rdpk3spfsal510.json", crate = crate);

define_low_storage_rk_from_file!(pub CKLLSRK43_2, "src/tableau/resources/low_storage/ckllsrk43-2.json", crate = crate);
define_low_storage_rk_from_file!(pub CKLLSRK54_3C, "src/tableau/resources/low_storage/ckllsrk54-3c.json", crate = crate);
define_low_storage_rk_from_file!(pub CKLLSRK95_4S, "src/tableau/resources/low_storage/ckllsrk95-4s.json", crate = crate);
define_low_storage_rk_from_file!(pub CKLLSRK95_4C, "src/tableau/resources/low_storage/ckllsrk95-4c.json", crate = crate);
define_low_storage_rk_from_file!(pub CKLLSRK95_4M, "src/tableau/resources/low_storage/ckllsrk95-4m.json", crate = crate);
define_low_storage_rk_from_file!(pub CKLLSRK54_3C_3R, "src/tableau/resources/low_storage/ckllsrk54-3c-3r.json", crate = crate);
define_low_storage_rk_from_file!(pub CKLLSRK54_3M_3R, "src/tableau/resources/low_storage/ckllsrk54-3m-3r.json", crate = crate);
define_low_storage_rk_from_file!(pub CKLLSRK54_3N_3R, "src/tableau/resources/low_storage/ckllsrk54-3n-3r.json", crate = crate);
define_low_storage_rk_from_file!(pub CKLLSRK85_4C_3R, "src/tableau/resources/low_storage/ckllsrk85-4c-3r.json", crate = crate);
define_low_storage_rk_from_file!(pub CKLLSRK85_4M_3R, "src/tableau/resources/low_storage/ckllsrk85-4m-3r.json", crate = crate);
define_low_storage_rk_from_file!(pub CKLLSRK85_4P_3R, "src/tableau/resources/low_storage/ckllsrk85-4p-3r.json", crate = crate);
define_low_storage_rk_from_file!(pub CKLLSRK54_3N_4R, "src/tableau/resources/low_storage/ckllsrk54-3n-4r.json", crate = crate);
define_low_storage_rk_from_file!(pub CKLLSRK54_3M_4R, "src/tableau/resources/low_storage/ckllsrk54-3m-4r.json", crate = crate);
define_low_storage_rk_from_file!(pub CKLLSRK65_4M_4R, "src/tableau/resources/low_storage/ckllsrk65-4m-4r.json", crate = crate);
define_low_storage_rk_from_file!(pub CKLLSRK85_4FM_4R, "src/tableau/resources/low_storage/ckllsrk85-4fm-4r.json", crate = crate);
define_low_storage_rk_from_file!(pub CKLLSRK75_4M_5R, "src/tableau/resources/low_storage/ckllsrk75-4m-5r.json", crate = crate);
