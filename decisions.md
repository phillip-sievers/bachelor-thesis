# Decisions

Any decisions that are made along the way documented

## Data pipeline

### Centroid

Matter less than i thought. We use distance signed to geometry for the labeling which eliminates the worry about false labels depending on the centorid strategy

### NAIP

Suspect Interplanetary Computer to be the winner but we need to test the hypothesis. See Notebook

Tested the planetary computer and works great. Wont try or explore via aws because reqires auth and i pay for network.
AWS rejection reason: .-9$/GB egress, so STAC index. USDA data.gov only allows full downloads -> more pain working with.
Rate limits is the better alternative to both imo

Accessing: STAC index query it -> naip collection with assests (images) -> sign in place (1hr TTL)
We can add a subscription key to get higher rate limits. No issue yet.

windowed read:

```
rasterio._io Jump straight to _read()
rasterio._io Window: Window(col_off=2091, row_off=453, width=512, height=512)
rasterio._io IO window xoff=2091.0 yoff=453.0 width=512.0 height=512.0
rasterio._err VSICURL: Downloading 123829135-125286130 (https://naipeuwest.blob.core.windows.net/naip/v002/ny/2022/ny_060cm_2022/40073/m_4007317_nw_18_060_20220719.tif?<token etc>)...
rasterio._err VSICURL: Downloading 137625156-139095429 (https://naipeuwest.blob.core.windows.net/naip/v002/ny/2022/ny_060cm_2022/40073/m_4007317_nw_18_060_20220719.tif?<token etc>)...
ny_m_4007317_nw_18_060_20220719 0.6 [12330, 9600]
```

Resolution: we pin GSD to 0.6m because only 24 states support 0.3m GSD. That leaves room for an optional experiment comparing .6 to .3 later on. however i suspect it doesnt change much because the model compresses the input image either way
THis resolves the fork at PHI-25 -> 512px is 307m

Year pin: Most recent 0.6m Imagery release per state -> minimal gap to Overture Data
We will still have to do some cleaning to verify temporal gap doesnt produce wrong labels ( built after image taken)

px and py need to saved to place pin correctly

### Noise

Distortion of high Buildings needs to be investigated (and how many high remote buildings there are)

### CRS

choosing to convert lat/longs from Geographic CRS to UTM. This splits USA into 6degrees each
EPSG:269xx

- minimal error
- matches NAD83 UTM by NAIP
- caveat: gotta pick the right zone per building -> zone lookup function

#### Additonal Info

Areas need to be computed in EPSG:5070 -> maintains exact areas
Distances as said above in UTM to match imagery and conformal

### Geometry

Rather than modyfying invalid footprints we filter and count the exclusions. This ensures correctness.
