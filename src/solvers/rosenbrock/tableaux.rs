//! Independently lazy Rosenbrock resources; no coefficient arrays are generated.

use crate::tableau::define_rosenbrock_tableau_from_file;

crate::tableau::define_rosenbrock_pair_tableau_from_file!(
    pub(super) ROSENBROCK23_32_TABLEAU,
    "Rosenbrock23/32",
    "src/tableau/resources/rosenbrock/rosenbrock23_32.json",
    crate = crate
);

define_rosenbrock_tableau_from_file!(pub(super) TSIT5DA_TABLEAU, "Tsit5DA",
    "src/tableau/resources/rosenbrock/tsit5da.json", crate = crate);

define_rosenbrock_tableau_from_file!(pub(super) ROS2_TABLEAU, "Ros2",
    "src/tableau/resources/rosenbrock/ros2.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) RODAS3_TABLEAU, "Rodas3",
    "src/tableau/resources/rosenbrock/rodas3.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) RODAS3D_TABLEAU, "Rodas3d",
    "src/tableau/resources/rosenbrock/rodas3d.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) ROS3_TABLEAU, "Ros3",
    "src/tableau/resources/rosenbrock/ros3.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) ROS3PR_TABLEAU, "Ros3Pr",
    "src/tableau/resources/rosenbrock/ros3pr.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) ROS3PRL_TABLEAU, "Ros3Prl",
    "src/tableau/resources/rosenbrock/ros3prl.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) ROS3PRL2_TABLEAU, "Ros3Prl2",
    "src/tableau/resources/rosenbrock/ros3prl2.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) ROS3P_TABLEAU, "Ros3p",
    "src/tableau/resources/rosenbrock/ros3p.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) ROS34PRW_TABLEAU, "Ros34Prw",
    "src/tableau/resources/rosenbrock/ros34prw.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) ROS34PW3_TABLEAU, "Ros34Pw3",
    "src/tableau/resources/rosenbrock/ros34pw3.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) GRK4A_TABLEAU, "Grk4a",
    "src/tableau/resources/rosenbrock/grk4a.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) GRK4T_TABLEAU, "Grk4t",
    "src/tableau/resources/rosenbrock/grk4t.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) ROK4A_TABLEAU, "Rok4a",
    "src/tableau/resources/rosenbrock/rok4a.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) ROS34PW1B_TABLEAU, "Ros34Pw1b",
    "src/tableau/resources/rosenbrock/ros34pw1b.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) ROS34PW2_TABLEAU, "Ros34Pw2",
    "src/tableau/resources/rosenbrock/ros34pw2.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) RODAS4_TABLEAU, "Rodas4",
    "src/tableau/resources/rosenbrock/rodas4.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) RODAS42_TABLEAU, "Rodas42",
    "src/tableau/resources/rosenbrock/rodas42.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) RODAS4P_TABLEAU, "Rodas4P",
    "src/tableau/resources/rosenbrock/rodas4p.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) RODAS4P2_TABLEAU, "Rodas4P2",
    "src/tableau/resources/rosenbrock/rodas4p2.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) RODAS4PW_TABLEAU, "Rodas4PW",
    "src/tableau/resources/rosenbrock/rodas4pw.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) RODAS5_TABLEAU, "Rodas5",
    "src/tableau/resources/rosenbrock/rodas5.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) RODAS5P_TABLEAU, "Rodas5P",
    "src/tableau/resources/rosenbrock/rodas5p.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) RODAS5PE_TABLEAU, "Rodas5Pe",
    "src/tableau/resources/rosenbrock/rodas5pe.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) RODAS6P_TABLEAU, "Rodas6P",
    "src/tableau/resources/rosenbrock/rodas6p.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) ROSENBROCK_W6S4OS_TABLEAU, "RosenbrockW6S4OS",
    "src/tableau/resources/rosenbrock/rosenbrock_w6s4os.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) RODAS23W_TABLEAU, "Rodas23W",
    "src/tableau/resources/rosenbrock/rodas23w.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) RODAS3P_TABLEAU, "Rodas3P",
    "src/tableau/resources/rosenbrock/rodas3p.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) ROS2PR_TABLEAU, "Ros2Pr",
    "src/tableau/resources/rosenbrock/ros2pr.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) ROS2S_TABLEAU, "Ros2S",
    "src/tableau/resources/rosenbrock/ros2s.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) ROS34PW1A_TABLEAU, "Ros34Pw1a",
    "src/tableau/resources/rosenbrock/ros34pw1a.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) ROS4LSTAB_TABLEAU, "Ros4LStab",
    "src/tableau/resources/rosenbrock/ros4lstab.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) ROSSHAMP4_TABLEAU, "RosShamp4",
    "src/tableau/resources/rosenbrock/rosshamp4.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) SCHOLZ4_7_TABLEAU, "Scholz4_7",
    "src/tableau/resources/rosenbrock/scholz4_7.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) VELDD4_TABLEAU, "Veldd4",
    "src/tableau/resources/rosenbrock/veldd4.json", crate = crate);
define_rosenbrock_tableau_from_file!(pub(super) VELDS4_TABLEAU, "Velds4",
    "src/tableau/resources/rosenbrock/velds4.json", crate = crate);
