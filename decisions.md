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
rasterio._io Window: Window(col_off=np.float64(2091.3484806375345), row_off=np.float64(453.2543646078557), width=np.float64(511.9999999999227), height=np.float64(511.9999999987583))
rasterio._io IO window xoff=2091.3484806375345 yoff=453.2543646078557 width=511.9999999999227 height=511.9999999987583
rasterio._err VSICURL: Downloading 123829135-125286130 (https://naipeuwest.blob.core.windows.net/naip/v002/ny/2022/ny_060cm_2022/40073/m_4007317_nw_18_060_20220719.tif?st=2026-09-01T15%3A06%3A49Z&se=2026-09-02T15%3A51%3A49Z&sp=rl&sv=2025-07-05&sr=c&skoid=9c8ff44a-6a2c-4dfb-b298-1c9212f64d9a&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2026-09-02T14%3A57%3A45Z&ske=2026-09-09T14%3A57%3A45Z&sks=b&skv=2025-07-05&sig=ox3iWzYnpxFCDtG8SeCARTvLUOZAk/Fqn5yfeJPlNbY%3D)...
rasterio._err VSICURL: Downloading 137625156-139095429 (https://naipeuwest.blob.core.windows.net/naip/v002/ny/2022/ny_060cm_2022/40073/m_4007317_nw_18_060_20220719.tif?st=2026-09-01T15%3A06%3A49Z&se=2026-09-02T15%3A51%3A49Z&sp=rl&sv=2025-07-05&sr=c&skoid=9c8ff44a-6a2c-4dfb-b298-1c9212f64d9a&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2026-09-02T14%3A57%3A45Z&ske=2026-09-09T14%3A57%3A45Z&sks=b&skv=2025-07-05&sig=ox3iWzYnpxFCDtG8SeCARTvLUOZAk/Fqn5yfeJPlNbY%3D)...
ny_m_4007317_nw_18_060_20220719 0.6 [12330, 9600]
```

Resolution: we pin GSD to 0.6m because only 24 states support 0.3m GSD. That leaves room for an optional experiment comparing .6 to .3 later on. however i suspect it doesnt change much because the model compresses the input image either way
THis resolves the fork at PHI-25 -> 512px is 307m

Year pin: Most recent 0.6m Imagery release per state -> minimal gap to Overture Data
We will still have to do some cleaning to verify temporal gap doesnt produce wrong labels ( built after image taken)
