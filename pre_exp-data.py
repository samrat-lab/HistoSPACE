import os
import pandas as pd
import scprep as scp
from tqdm import tqdm


# Meta variables #update all the paths based on STNet dataset
RAW_PATH = '../Input/raw/HBC_stnet/'
IMG_PATH = '../Input/raw/HBC_stnet/images/'
CM_PATH = '../Input/raw/HBC_stnet/count_matrix/'
SPOT_PATH = '../Input/raw/HBC_stnet/spots/'
COORD_PATH = '../Input/raw/HBC_stnet/tumor_coord/'
META_PATH = '../Input/raw/HBC_stnet/metadata.csv'
TEMPLATE_IMG = 'Provide_a_template_path_for_img_normalization.jpg'
no_of_pats = 23

# Function to add labels from metadata
def add_label(dataframe, label, meta):
    label_list = []
    for spot in dataframe.index.values:
        sample_id = '_'.join(spot.split('_')[:-1])
        spot_label = meta.loc[sample_id, label]
        label_list.append(spot_label)
    dataframe[label] = label_list
    return dataframe

# Function to subset a DataFrame based on index
def subset_df(df, indx):
    com_indx = list(set(df.index) & set(indx)) 
    return df.loc[com_indx]

# Function to normalize and rename count matrix
def get_norm_df(cnt, ensg_dict):
    full_norm = scp.transform.log(scp.normalize.library_size_normalize(cnt.values))
    norm_cnt = pd.DataFrame(data=full_norm, index=cnt.index, columns=cnt.columns)
    return norm_cnt.rename(columns=ensg_dict)

# Ensure that paths are correct and files exist before loading
ensg_map_path = 'ensembl.tsv'
assert os.path.exists(ensg_map_path), f"File not found: {ensg_map_path}"  
ensg_map = pd.read_csv(ensg_map_path, sep='\t')
ensg_dict = ensg_map.set_index('Ensembl ID(supplied by Ensembl)')['Approved symbol'].to_dict()

meta_data = pd.read_csv(META_PATH)
appended_data = []

# Iterate through metadata to process count matrices and spot files
for _id, item in tqdm(meta_data.iterrows()):
    cm_file = CM_PATH + item['count_matrix'][:-3]
    if not os.path.exists(cm_file): 
        print(f"Missing count matrix file: {cm_file}")
        continue

    cm_data = pd.read_csv(cm_file, sep='\t', index_col=0)
    temp_cm = get_norm_df(cm_data, ensg_dict)

    spot_file = SPOT_PATH + item['spot_coordinates'][:-3]
    if not os.path.exists(spot_file): 
        print(f"Missing spot file: {spot_file}")
        continue

    spot_data = pd.read_csv(spot_file, index_col=0)
    spot_indx = spot_data.index.to_list()
    cm = subset_df(temp_cm, spot_indx)

    sample_n = '_'.join(os.path.basename(cm_file).split("_")[:-1])
    new_spots = [f"{sample_n}_{spot}" for spot in cm.index]
    cm.index = new_spots

    tumor_loc_file = COORD_PATH + item['tumor_annotation'][:-3]
    if not os.path.exists(tumor_loc_file):
        print(f"Missing tumor annotation file: {tumor_loc_file}")
        continue

    tumor_loc = pd.read_csv(tumor_loc_file, sep='\t')
    tumor_indx = [f"{sample_n}_{int(y)}x{int(lab)}" for y, lab in zip(tumor_loc['ycoord'].round(0), tumor_loc['lab'].round(0))]
    tumor_loc.index = tumor_indx
    tumor_loc.rename(columns={'Unnamed: 4': 'tumor_status'}, inplace=True)
    cm_new = cm.join(tumor_loc['tumor_status'])

    appended_data.append(cm_new)

# Concatenate all data frames
total_counts = pd.concat(appended_data)
print(total_counts.shape)

# Filter out columns with missing values
my_cols = ~total_counts.isna().any()
my_cols['tumor_status'] = True 
temp_df = total_counts.loc[:, my_cols]

# Filter genes and samples based on some conditions
exp_genes = list(my_values[my_values < 0.5].keys())
sub_exp_df = temp_df[exp_genes]

exp_samples = list(my_values[my_values < 0.5].keys())
new_exp = sub_exp_df.loc[exp_samples]

# Remove columns containing 'ENSG'
temp2 = new_exp.filter(regex=r'^(?!.*ENSG).*$', axis=1)

# Round the final dataframe and save to file
final_df = temp2.round(3)
final_df.to_csv(TILE_PATH + 'count_mat(top_gene_0.5).tsv', sep='\t')

# Create a mapping for patients and assign folds for cross-validation
path_df = final_df.copy()
path_df['Patients'] = path_df.index.str.replace(r'_(.*)', '', regex=True)  # {DEBUG 6}: Corrected regex
pat_map = dict(zip(path_df['Patients'].unique(), range(num_of_pats)))
path_df['fold'] = path_df['Patients'].map(pat_map)

# Save the fold dataset
fold_df_path = 'file_location_and_name.csv' # preferably: dataset_folds(top_gene_0.5).csv used in downstream scripts'
path_df.to_csv(fold_df_path)
