from objectbox.model import *
from objectbox.model.properties import Index, IndexType

# Auto-generated ObjectBox entities with UIDs from Kotlin
# Generated from: database/default.json


@Entity(uid=3616035618583444316)
class VectorEntity1024:
    id = Id(id=1, uid=2660901179653327145)
    name = String(id=2, uid=6856925769981232309)
    content = String(id=3, uid=179843414054036048, index=Index(type=IndexType.VALUE, uid=7296155417454871405))
    embedding = Float32Vector(
        id=4,
        uid=369938825836818176,
        index=HnswIndex(dimensions=1024, distance_type=VectorDistanceType.COSINE),
    )
    metadata = String(id=5, uid=4249614817927516900)
    timestamp = Property(int, id=6, uid=5216177833509211649)
    document = String(id=7, uid=8187095563063126218)


@Entity(uid=2070669380000345579)
class VectorEntity128:
    id = Id(id=1, uid=3440642330255188291)
    name = String(id=2, uid=7132069898043676772)
    content = String(
        id=3, uid=1078614701445763727, index=Index(type=IndexType.VALUE, uid=1703725148714040606)
    )
    embedding = Float32Vector(
        id=4,
        uid=7560827336622966584,
        index=HnswIndex(dimensions=128, distance_type=VectorDistanceType.COSINE),
    )
    metadata = String(id=5, uid=1578594461034605409)
    timestamp = Property(int, id=6, uid=5209582188171579796)
    document = String(id=7, uid=1746313554083619572)


@Entity(uid=7000709663387574396)
class VectorEntity1536:
    id = Id(id=1, uid=5191340093058569025)
    name = String(id=2, uid=5533256385720403749)
    content = String(
        id=3, uid=4233813361513752915, index=Index(type=IndexType.VALUE, uid=4136246846239337743)
    )
    embedding = Float32Vector(
        id=4,
        uid=6332384653779176856,
        index=HnswIndex(dimensions=1536, distance_type=VectorDistanceType.COSINE),
    )
    metadata = String(id=5, uid=6594569271998485886)
    timestamp = Property(int, id=6, uid=8326824004865985967)
    document = String(id=7, uid=8395961762390350544)


@Entity(uid=5228878220447563421)
class VectorEntity256:
    id = Id(id=1, uid=3644592590890843602)
    name = String(id=2, uid=1965666759894323807)
    content = String(
        id=3, uid=8662363688997292720, index=Index(type=IndexType.VALUE, uid=9111919132284886667)
    )
    embedding = Float32Vector(
        id=4,
        uid=4216566114920401960,
        index=HnswIndex(dimensions=256, distance_type=VectorDistanceType.COSINE),
    )
    metadata = String(id=5, uid=1867185820949922463)
    timestamp = Property(int, id=6, uid=4768130956928245718)
    document = String(id=7, uid=4421787786747795333)


@Entity(uid=5759397291334530001)
class VectorEntity384:
    id = Id(id=1, uid=3995752191779281531)
    name = String(id=2, uid=8429658584629108761)
    content = String(
        id=3, uid=7177388378640589383, index=Index(type=IndexType.VALUE, uid=5351510252118108820)
    )
    embedding = Float32Vector(
        id=4,
        uid=7475465692145108710,
        index=HnswIndex(dimensions=384, distance_type=VectorDistanceType.COSINE),
    )
    metadata = String(id=5, uid=693547862910235008)
    timestamp = Property(int, id=6, uid=8172946629643359289)
    document = String(id=7, uid=7209720699072026712)


@Entity(uid=5741478164101520808)
class VectorEntity512:
    id = Id(id=1, uid=2928259287888584009)
    name = String(id=2, uid=3586580055118903977)
    content = String(
        id=3, uid=3705505580070095385, index=Index(type=IndexType.VALUE, uid=2382896421061802420)
    )
    embedding = Float32Vector(
        id=4,
        uid=8704612602687791410,
        index=HnswIndex(dimensions=512, distance_type=VectorDistanceType.COSINE),
    )
    metadata = String(id=5, uid=1811001360407727355)
    timestamp = Property(int, id=6, uid=8246025844161259468)
    document = String(id=7, uid=2415562500626563670)


@Entity(uid=5652731347474038494)
class VectorEntity64:
    id = Id(id=1, uid=8122553010226434725)
    name = String(id=2, uid=2970097330866361593)
    content = String(
        id=3, uid=3234213945405272885, index=Index(type=IndexType.VALUE, uid=7169805229422572048)
    )
    embedding = Float32Vector(
        id=4, uid=1702504652196090376, index=HnswIndex(dimensions=64, distance_type=VectorDistanceType.COSINE)
    )
    metadata = String(id=5, uid=1325344924173597267)
    timestamp = Property(int, id=6, uid=8447383474601691598)
    document = String(id=7, uid=8146090976975030625)


@Entity(uid=1398842727202970551)
class VectorEntity768:
    id = Id(id=1, uid=1573037040010322675)
    name = String(id=2, uid=1875010556206429126)
    content = String(id=3, uid=4026475639719700418, index=Index(type=IndexType.VALUE, uid=443802177034598622))
    embedding = Float32Vector(
        id=4,
        uid=7215409883456044223,
        index=HnswIndex(dimensions=768, distance_type=VectorDistanceType.COSINE),
    )
    metadata = String(id=5, uid=5797340244742986745)
    timestamp = Property(int, id=6, uid=5763104868674049307)
    document = String(id=7, uid=7290423554697741641)
