"""Proof-of-concept settings: one crop, one region, fixed time splits."""

# Conilon (robusta) municipalities in northern Espírito Santo, the main
# conilon region and strongly family-farmed. PAM's arabica/canephora split only
# starts in 2012, so the PoC uses "café total", which here is mostly conilon.
# Rondônia is left out for now: its clonal-variety boom (yields roughly
# tripled after ~2013) needs its own trend handling.
MUNICIPIOS = [
    ("Jaguaré", "ES"), ("Linhares", "ES"), ("Vila Valério", "ES"),
    ("São Gabriel da Palha", "ES"), ("Nova Venécia", "ES"), ("Sooretama", "ES"),
    ("Rio Bananal", "ES"), ("São Mateus", "ES"), ("Pinheiros", "ES"),
    ("Águia Branca", "ES"), ("Governador Lindenberg", "ES"), ("Colatina", "ES"),
    ("Marilândia", "ES"), ("São Domingos do Norte", "ES"), ("Vila Pavão", "ES"),
    ("Barra de São Francisco", "ES"), ("Boa Esperança", "ES"), ("Montanha", "ES"),
]
PRODUCT = "cafe_total"
PRICE_SERIES = "wb:coffee_robusta"   # USD/kg; converted to BRL/saca with PTAX

# PAM harvest years used. Earlier years are skipped: coffee units and
# methodology changed around 2001, so the series is not comparable before.
FIRST_YEAR = 2002

# Yield model splits (by harvest year, never shuffled)
YIELD_TRAIN_END = 2014      # 2002-2014 train
YIELD_CAL_END = 2019        # 2015-2019 calibration
# 2020-2025 test (includes the 2021 frost and drought -> 2022 crop)

# Price model: monthly BRL/saca from this month on (floating exchange rate)
PRICE_START = "1999-03-01"
PRICE_HORIZON = 6            # months ahead, roughly planting/flowering -> harvest
PRICE_TRAIN_END = "2015-12-01"
PRICE_CAL_END = "2019-12-01"

ALPHA = 0.2                  # 80% intervals
