import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(dataset_dir):
    train_set = pd.read_parquet(dataset_dir+"train.parquet")
    valid_set = pd.read_parquet(dataset_dir+"valid.parquet")
    test_set = pd.read_parquet(dataset_dir+"test.parquet")

    return train_set, valid_set, test_set

def merge_inferenece(df, df_compare):
### BELOW CODE IS FOR MERGING INFERENCE RESULTS WITH THEIR INFORMATION

    # print(df_compare)
    # df_compare = df_compare.drop('Unnamed: 0', axis=1)

    df_new = tensor_id_merge(df, df_compare)
    return df_new
    
def tensor_id_merge(df1, df2):
    df3 = pd.merge(df1, df2, on='tensor id', how='inner')

    return df3

def check_labels_count(df):
    print("checking count for each labels...\n-------")
    print(pd.Series(df["label"]).value_counts().reset_index())

def preds_check_labels_count(df):
    print("checking count for each labels...\n-------")
    print(pd.Series(df["max label"]).value_counts().reset_index())
    
def matching_labels(df):
    print("checking matching prediction and base results...'n")
    matching_labels = df["max label"] == df["label"]
    print(matching_labels)
    df["matching label"] = matching_labels
    
    return df
    
# def dataset_split(df, save_dir):
#     train_set, test_set = train_test_split(df, test_size=0.3, random_state=42, stratify=
#     df["label"].values.tolist())
#     val_set, test_set = train_test_split(test_set, test_size=0.333, random_state=42, stratify=
#     test_set["label"].values.tolist())
#     my_train = train_set[["sequence", "label", "tensor id"]]
#     my_valid = val_set[["sequence", "label", "tensor id"]]
#     my_test = test_set[["sequence", "label", "tensor id"]]

#     my_train.to_parquet(save_dir + "train.parquet")
#     check_labels_count(my_train)
#     my_valid.to_parquet(save_dir + "valid.parquet")
#     check_labels_count(my_valid)
#     my_test.to_parquet(save_dir + "test.parquet")
#     check_labels_count(my_test)
      
def join_train_data(path):
    df_train = pd.read_parquet(path+"train.parquet")
    df_valid = pd.read_parquet(path+"valid.parquet")
    df_test = pd.read_parquet(path+"test.parquet")
    df = pd.concat([df_train, df_valid, df_test])
    
    return df

def add_train_data(path, save_path, sample_df):
    df_train_ori = pd.read_parquet(path+"train.parquet")
    df_valid_ori = pd.read_parquet(path+"valid.parquet")
    df_test_ori = pd.read_parquet(path+"test.parquet")
    
    
    train_set, test_set = train_test_split(sample_df, test_size=0.3, random_state=42, stratify=
    sample_df["label"].values.tolist())
    val_set, test_set = train_test_split(test_set, test_size=0.333, random_state=42, stratify=
    test_set["label"].values.tolist())
    
    my_train = train_set[["sequence", "label", "tensor id"]]
    my_valid = val_set[["sequence", "label", "tensor id"]]
    my_test = test_set[["sequence", "label", "tensor id"]]

    # my_train = give_tensor_id(pd.concat([df_train_ori, my_train]))
    # my_valid = give_tensor_id(pd.concat([df_valid_ori, my_valid]))
    # my_test = give_tensor_id(pd.concat([df_test_ori, my_test]))
    my_train = pd.concat([df_train_ori, my_train])
    my_valid = pd.concat([df_valid_ori, my_valid])
    my_test = pd.concat([df_test_ori, my_test])

    my_train.to_parquet(save_path + "train.parquet", index = False)
    check_labels_count(my_train)
    my_valid.to_parquet(save_path + "valid.parquet", index = False)
    check_labels_count(my_valid)
    my_test.to_parquet(save_path + "test.parquet", index = False)
    check_labels_count(my_test)
      
def drop_duplicates(df, df_compare):
    # Column to use for matching
    column_to_match = 'tensor id'
    # Merge the DataFrames on the specified column
    merged = pd.merge(df, df_compare, on=column_to_match, how='inner')
    # print(df)
    # print(df_compare)
    # print(merged)
    # Filter out the rows from df_b that are duplicates of rows in df_a based on the specified column
    result = df[~df[column_to_match].isin(merged[column_to_match])]
    # print(result)
    return result


def get_group_safely(grouped, group_name):
    try:
        group = grouped.get_group(group_name)
        return group
    except KeyError:
        print(f"Group '{group_name}' not found")
        return pd.DataFrame()
        
# def sample_trainable_df_5class(df, sample_count):
#     df_by_labels = df.groupby(df.label)
#     #dictionary for identifying which class is empty and how much data it has left
#     empty_class_dict = {}
#     normal_df = get_group_safely(df_by_labels,0)
#     secretion_df = get_group_safely(df_by_labels,1)
#     resistant_df = get_group_safely(df_by_labels,2)
#     toxin_df = get_group_safely(df_by_labels,3)
#     anti_toxin_df = get_group_safely(df_by_labels,4)
#     virulence_df= get_group_safely(df_by_labels,5)
    
#     if len(normal_df)>=sample_count:
#         normal_random_df=normal_df.sample(sample_count)
#         empty_class_dict["normal"] = 1
#     else:
#         normal_random_df = normal_df.sample(len(normal_df))
#         empty_class_dict["normal"] = len(normal_df)
        
#     if len(secretion_df)>=sample_count:
#         secretion_random_df = secretion_df.sample(sample_count)
#         empty_class_dict["secretion"] = 1
#     else:
#         secretion_random_df = secretion_df.sample(len(secretion_df))
#         empty_class_dict["secretion"] = len(secretion_df)
        
#     if len(resistant_df)>=sample_count:
#         resistant_random_df = resistant_df.sample(sample_count)
#         empty_class_dict["resistant"] = 1
#     else:
#         resistant_random_df = resistant_df.sample(len(resistant_df))
#         empty_class_dict["resistant"] = len(resistant_df)
        
#     if len(toxin_df)>=sample_count:
#         toxin_random_df = toxin_df.sample(sample_count)
#         empty_class_dict["toxin"] = 1
#     else:
#         toxin_random_df = toxin_df.sample(len(toxin_df))
#         empty_class_dict["toxin"] = len(toxin_df)
        
#     if len(anti_toxin_df)>=sample_count:
#         anti_toxin_random_df = anti_toxin_df.sample(sample_count)
#         empty_class_dict["anti_toxin"] = 1
#     else:
#         anti_toxin_random_df = anti_toxin_df.sample(len(anti_toxin_df))
#         empty_class_dict["anti_toxin"] = len(anti_toxin_df)
        
        
#     if len(virulence_df)>=sample_count:
#         virulence_random_df = virulence_df.sample(sample_count)
#         empty_class_dict["virulence"] = 1
#     else:
#         virulence_random_df = virulence_df.sample(len(virulence_df))
#         empty_class_dict["virulence"] = len(virulence_df)       
    
        
#     sample_df = pd.concat([normal_random_df, secretion_random_df, resistant_random_df, toxin_random_df, anti_toxin_random_df, virulence_random_df])
    
#     return sample_df, empty_class_dict

   

def add_preds_data(df, preds_df, sample_count, empty_class_dict):
    
    df_by_labels = preds_df.groupby(preds_df.label)
    secretion_df = get_group_safely(df_by_labels,1)
    resistant_df = get_group_safely(df_by_labels,2)
    toxin_df = get_group_safely(df_by_labels,3)
    anti_toxin_df = get_group_safely(df_by_labels,4)
    virulence_df= get_group_safely(df_by_labels,5)
    
    if empty_class_dict["secretion"] == 0:
        secretion_random_df = secretion_df.sample(sample_count)
    elif empty_class_dict["secretion"] == 1:
        secretion_random_df = secretion_df.sample(0)
    else:
        num_to_add = sample_count - int(empty_class_dict["secretion"])
        secretion_random_df = secretion_df.sample(num_to_add)
    
    if empty_class_dict["resistant"] == 0:
        resistant_random_df = resistant_df.sample(sample_count)
    elif empty_class_dict["resistant"] == 1:
        resistant_random_df = resistant_df.sample(0)
    else:
        num_to_add = sample_count - int(empty_class_dict["resistant"])
        resistant_random_df = resistant_df.sample(num_to_add)

    if empty_class_dict["toxin"] == 0:
        toxin_random_df = toxin_df.sample(sample_count)
    elif empty_class_dict["toxin"] == 1:
        toxin_random_df = toxin_df.sample(0)
    else:
        num_to_add = sample_count - int(empty_class_dict["toxin"])
        toxin_random_df = toxin_df.sample(num_to_add)
        
    if empty_class_dict["virulence"] == 0:
        virulence_random_df = virulence_df.sample(sample_count)
    elif empty_class_dict["virulence"] == 1:
        virulence_random_df = virulence_df.sample(0)
    else:
        num_to_add = sample_count - int(empty_class_dict["virulence"])
        virulence_random_df = virulence_df.sample(num_to_add)
        
    if empty_class_dict["anti_toxin"] == 0:
        anti_toxin_random_df = anti_toxin_df.sample(sample_count)
    elif empty_class_dict["anti_toxin"] == 1:
        anti_toxin_random_df = anti_toxin_df.sample(0)
    else:
        num_to_add = sample_count - int(empty_class_dict["anti_toxin"])
        anti_toxin_random_df = anti_toxin_df.sample(num_to_add)
        


    sample_df = pd.concat([df, secretion_random_df,resistant_random_df, virulence_random_df, toxin_random_df, anti_toxin_random_df])
    
    return sample_df


def give_tensor_id(df):
    index_no = []
    for i in range(len(df["sequence"])):
       index_no.append(i)
    df["tensor id"] = index_no
    
    return df


# splits df to train, validation and test sets
def dataset_split(swiss_df,trembl_df, save_dir, n_samples_per_label):


    # Sample n_samples_per_label for each label. If swiss_df has fewer samples for a label, supplement from trembl_df.
    final_samples = []
    swiss_samples_for_count = []
    trembl_samples_for_count = []
    for label in swiss_df['label'].unique():
        swiss_group = swiss_df[swiss_df['label'] == label]

        if len(swiss_group) >= n_samples_per_label:
            swiss_samples = swiss_group.sample(n=n_samples_per_label, random_state=42)
            final_samples.append(swiss_samples)
            swiss_samples_for_count.append(swiss_samples)
        else:
            # Take all from swiss
            swiss_samples = swiss_group
            swiss_samples_for_count.append(swiss_samples)
            needed = n_samples_per_label - len(swiss_samples)

            # Get samples from trembl
            if label in trembl_df['label'].unique():
                trembl_group = trembl_df[trembl_df['label'] == label]
                # avoid duplicates
                trembl_group = trembl_group[~trembl_group['Sequence'].isin(swiss_samples['Sequence'])]

                num_to_sample = min(needed, len(trembl_group))
                trembl_samples = trembl_group.sample(n=num_to_sample, random_state=42)
                trembl_samples_for_count.append(trembl_samples)

                final_samples.append(pd.concat([swiss_samples, trembl_samples]))
            else:
                final_samples.append(swiss_samples)

    print("--- Samples drawn from swiss_df ---")
    if swiss_samples_for_count:
        check_labels_count(pd.concat(swiss_samples_for_count))
    else:
        print("No samples drawn from swiss_df.")

    print("\n--- Samples drawn from trembl_df ---")
    if trembl_samples_for_count:
        check_labels_count(pd.concat(trembl_samples_for_count))
    else:
        print("No samples drawn from trembl_df.")

    sampled_df = pd.concat(final_samples).reset_index(drop=True)
    print("\n--- Total samples in final dataset ---")
    check_labels_count(sampled_df)

    train_set, test_set = train_test_split(sampled_df, test_size=0.3, random_state=42, stratify=
    sampled_df["label"].values.tolist())
    val_set, test_set = train_test_split(test_set, test_size=0.333, random_state=42, stratify=
    test_set["label"].values.tolist())
    my_train = train_set[["Sequence", "label", "tensor id"]]
    my_valid = val_set[["Sequence", "label", "tensor id"]]
    my_test = test_set[["Sequence", "label", "tensor id"]]

    my_train.to_parquet(save_dir + "train.parquet", index = False)
    check_labels_count(my_train)
    my_valid.to_parquet(save_dir + "valid.parquet", index = False)
    check_labels_count(my_valid)
    my_test.to_parquet(save_dir + "test.parquet", index = False)
    check_labels_count(my_test)