"""APG IV taxonomy (order > family > genus > species) for the 30 species.

Keys are the dataset folder names. Placement follows APG IV (2016); see paper Table S1
for per-species sources (POWO / Plants of the World Online).
"""
from __future__ import annotations

TAXONOMY: dict[str, tuple[str, str, str, str]] = {
    # folder name: (order, family, genus, species epithet as used in paper)
    "Alpinia Galanga (Rasna)": ("Zingiberales", "Zingiberaceae", "Alpinia", "Alpinia galanga"),
    "Amaranthus Viridis (Arive-Dantu)": ("Caryophyllales", "Amaranthaceae", "Amaranthus", "Amaranthus viridis"),
    "Artocarpus Heterophyllus (Jackfruit)": ("Rosales", "Moraceae", "Artocarpus", "Artocarpus heterophyllus"),
    "Azadirachta Indica (Neem)": ("Sapindales", "Meliaceae", "Azadirachta", "Azadirachta indica"),
    "Basella Alba (Basale)": ("Caryophyllales", "Basellaceae", "Basella", "Basella alba"),
    "Brassica Juncea (Indian Mustard)": ("Brassicales", "Brassicaceae", "Brassica", "Brassica juncea"),
    "Carissa Carandas (Karanda)": ("Gentianales", "Apocynaceae", "Carissa", "Carissa carandas"),
    "Citrus Limon (Lemon)": ("Sapindales", "Rutaceae", "Citrus", "Citrus limon"),
    "Ficus Auriculata (Roxburgh fig)": ("Rosales", "Moraceae", "Ficus", "Ficus auriculata"),
    "Ficus Religiosa (Peepal Tree)": ("Rosales", "Moraceae", "Ficus", "Ficus religiosa"),
    "Hibiscus Rosa-sinensis": ("Malvales", "Malvaceae", "Hibiscus", "Hibiscus rosa-sinensis"),
    "Jasminum (Jasmine)": ("Lamiales", "Oleaceae", "Jasminum", "Jasminum sp."),
    "Mangifera Indica (Mango)": ("Sapindales", "Anacardiaceae", "Mangifera", "Mangifera indica"),
    "Mentha (Mint)": ("Lamiales", "Lamiaceae", "Mentha", "Mentha sp."),
    "Moringa Oleifera (Drumstick)": ("Brassicales", "Moringaceae", "Moringa", "Moringa oleifera"),
    "Muntingia Calabura (Jamaica Cherry-Gasagase)": ("Malvales", "Muntingiaceae", "Muntingia", "Muntingia calabura"),
    "Murraya Koenigii (Curry)": ("Sapindales", "Rutaceae", "Murraya", "Murraya koenigii"),
    "Nerium Oleander (Oleander)": ("Gentianales", "Apocynaceae", "Nerium", "Nerium oleander"),
    "Nyctanthes Arbor-tristis (Parijata)": ("Lamiales", "Oleaceae", "Nyctanthes", "Nyctanthes arbor-tristis"),
    "Ocimum Tenuiflorum (Tulsi)": ("Lamiales", "Lamiaceae", "Ocimum", "Ocimum tenuiflorum"),
    "Piper Betle (Betel)": ("Piperales", "Piperaceae", "Piper", "Piper betle"),
    "Plectranthus Amboinicus (Mexican Mint)": ("Lamiales", "Lamiaceae", "Plectranthus", "Plectranthus amboinicus"),
    "Pongamia Pinnata (Indian Beech)": ("Fabales", "Fabaceae", "Pongamia", "Pongamia pinnata"),
    "Psidium Guajava (Guava)": ("Myrtales", "Myrtaceae", "Psidium", "Psidium guajava"),
    "Punica Granatum (Pomegranate)": ("Myrtales", "Lythraceae", "Punica", "Punica granatum"),
    "Santalum Album (Sandalwood)": ("Santalales", "Santalaceae", "Santalum", "Santalum album"),
    "Syzygium Cumini (Jamun)": ("Myrtales", "Myrtaceae", "Syzygium", "Syzygium cumini"),
    "Syzygium Jambos (Rose Apple)": ("Myrtales", "Myrtaceae", "Syzygium", "Syzygium jambos"),
    "Tabernaemontana Divaricata (Crape Jasmine)": ("Gentianales", "Apocynaceae", "Tabernaemontana", "Tabernaemontana divaricata"),
    "Trigonella Foenum-graecum (Fenugreek)": ("Fabales", "Fabaceae", "Trigonella", "Trigonella foenum-graecum"),
}

LEVELS = ("order", "family", "genus", "species")
CLASSES = sorted(TAXONOMY)  # canonical class order == sorted folder names


def level_labels(level: str) -> tuple[list[str], list[int]]:
    """Return (names at that level, mapping species index -> level index)."""
    li = LEVELS.index(level)
    names = sorted({TAXONOMY[c][li] for c in CLASSES})
    return names, [names.index(TAXONOMY[c][li]) for c in CLASSES]


def tree_distance(i: int, j: int) -> int:
    """Number of levels (0..4) up to the lowest common ancestor; 0 = same species."""
    if i == j:
        return 0
    a, b = TAXONOMY[CLASSES[i]], TAXONOMY[CLASSES[j]]
    for depth in (2, 1, 0):  # genus, family, order
        if a[depth] == b[depth]:
            return 3 - depth
    return 4
