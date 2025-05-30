from pytket.architecture import Architecture
from pytket.backends.backendinfo import BackendInfo
from pytket.circuit import OpType  # type: ignore

mahattan_link = [(0,10),(0,1),(1,0),(1,2),(2,1),(2,3),(3,4),(3,2),(4,3),(4,5),(4,11),(5,4),(5,6),(6,5),(6,7),(7,6),(7,8),(8,9),(8,12),(8,7),(9,8),(10,0),(10,13),(11,4),(11,17),(12,21),(12,8),(13,10),(13,14),(14,15),(14,13),(15,14),(15,16),(15,24),(16,15),(16,17),(17,16),(17,18),(17,11),(18,19),(18,17),(19,20),(19,18),(19,25),(20,21),(20,19),(21,12),(21,22),(21,20),(22,21),(22,23),(23,26),(23,22),(24,29),(24,15),(25,33),(25,19),(26,23),(26,37),(27,38),(27,28),(28,29),(28,27),(29,24),(29,28),(29,30),(30,31),(30,29),(31,30),(31,32),(31,39),(32,31),(32,33),(33,32),(33,25),(33,34),(34,35),(34,33),(35,34),(35,40),(35,36),(36,37),(36,35),(37,26),(37,36),(38,41),(38,27),(39,45),(39,31),(40,49),(40,35),(41,38),(41,42),(42,41),(42,43),(43,42),(43,44),(43,52),(44,43),(44,45),(45,44),(45,39),(45,46),(46,47),(46,45),(47,48),(47,53),(47,46),(48,47),(48,49),(49,40),(49,50),(49,48),(50,51),(50,49),(51,50),(51,54),(52,43),(52,56),(53,47),(53,60),(54,64),(54,51),(55,56),(56,57),(56,55),(56,52),(57,56),(57,58),(58,57),(58,59),(59,60),(59,58),(60,53),(60,59),(60,61),(61,62),(61,60),(62,61),(62,63),(63,62),(63,64),(64,54),(64,63)]
mahattan_coupling = [list(i) for i in mahattan_link]
mahattan_arch = Architecture(mahattan_coupling)
mahattan_device = BackendInfo(
    name='my_backend', device_name='my_device', version='1.0',
    architecture=mahattan_arch,
    gate_set={
        OpType.Rx,
        OpType.Ry,
        OpType.Rz,
        OpType.U3,
        OpType.CX,
        OpType.H,
        OpType.X,
        OpType.Y,
        OpType.S,
        OpType.Sdg,
        OpType.Measure
    })

