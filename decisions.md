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

For edges we decided to discard edges over stitching mosaic images together. This is to ensure images are consistent for a single address.

We also always pick the first of all image candidates. They overlap. I wont implement sampling logic that checks all candidates for the most center/best fitting window. The first candidate -> test -> reject if it doesnt fit path will still give us more than plenty rural candidates and houses for the dataset

The error rate with this picking stretgy lies at approx 3%

### Cache decisions

cache key will be (stac_item_id, col_off, row_off, size) this means we wont be susceptible to instable floats if we cached on (lat, lon, gsd, ...)

We drop the NIR band, we dont need it, only RGB and store JSON pre-render. pin rendering embedded means the ablations of pin costs fetches.

We alsways select the most recent 0.6m GSD this is deterministic per state and means the dataset is repro

We wont resample or tamper with resolutions. We always require GSD 0.6m

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

### Define Rural and sampling

Rural means the complement of the 2020 Census urban-area delineation, using the pinned 2025 TIGER `tl_2025_us_uac20.zip` artifact. Each 307.2 m image is centred on the moved pin. Full extent plus a 100 m margin must be disjoint from all urban areas. We keep valid footprints with area over 100 m2.

The full target footprint and original anchor must remain in the image. Off-target pins that hit another Overture footprint are resampled. This removes known ambiguous negatives without selecting for sparse footprint coverage

visible-roof disagreement and missing footprints remain for manual QA

### Marker

We will use three shaopes and three colors and run prompt ablations to test each variable . 9 combinations total. small ablations before the big runs.

The red_dot seems to be best based on the eye test across three different surfaces. Itll be the provisional default. A final choice will happen during ablations.

## Models

### PHI-39 — selection frozen 2026-09-04

Selected `gpt-5.6-sol`, `claude-opus-5`, and `gemini-3.6-flash`

That means one frontier class of models across three providers, so three different families.

The problem with open models is lack of hosting providers, the same issue applies to RS Model specialists.

We lock model ID and all reasoning efforts set to 'low'. This however doesnt ensure the same reasoning effort across all three models. Thats why we also save the reasoning tokens etc.

#### No open weights?

Open weights does not make the experiments more reproducable. It only makes the results auditable. This study never set out to analyse the internals of the models and how they came up with the results. We solely try to establish a benchmark that lays the groundwork for agentic work of agents on geospatial imagery. The three comparable frontier model are a much more interesting case for this than smaller, less comparable open models. Furthermore the open models require very controlled and equal hardware, which i would be equally bound to by a provider sincei. dont own the needed hardware.

This however leaves a explicit limitation: Observations generalize only to the providers and their model families, not all VLMs or pradigms

#### No RS specialist model?

Including a fine tuned model specifcially on RS data would introduce many variables that leaves us unable to attribute differences to the model: task prior, architecture, model scale, input resolution etc. Furthermore is a general RS model not trained on this specific task either.

... more stuff in overleaf


## Sampling

We need 2100 unique locations. 100 of which will be used on a held out dataset to run marker and prompt ablations.

Was a test where i just ran the candidate building
```
python src/thesis/run.py
RetentionStage(name='input', count=1377087, fraction_of_input=1.0, fraction_of_previous=1.0)
RetentionStage(name='valid_polygon', count=1377008, fraction_of_input=0.9999426325279376, fraction_of_previous=0.9999426325279376)
RetentionStage(name='min_area_m2', count=1003510, fraction_of_input=0.7287193910043447, fraction_of_previous=0.7287611981920221)
RetentionStage(name='max_aspect_ratio', count=1000497, fraction_of_input=0.7265314391901165, fraction_of_previous=0.9969975386393758)
RetentionStage(name='census_rural', count=421080, fraction_of_input=0.3057758877979387, fraction_of_previous=0.42087082719888214)
```

the boxes are fine if near cities. Waht matters is the houses. i think that makes the distirbution and eval more fair than the whole 25km being rural, that leaves only super duper rural places.
Boxes distribution: it reflects real differences in how much eligible rural land exists per ecoregion.

# Results

We parse for a yes/no and discard if more than one foound. Furthermore we report refusals/errors separaetly. Outside of recall/precision. This means recall and precision might be inflated if model refuses on hard cases but the separate report fixes that.

We dont yet need resampling. If at a lter point we want to replay over same samples for analysing determinism then we need to change the cache keys and implement an additonal sampling index or smth
