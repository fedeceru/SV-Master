# Carica i dati dai file CSV e srestituisce i rispettivi DataFrame.

import pandas as pd
import os
from typing import Tuple

paths = {
    "BP":  "gene_go_matrix_propT_rel-is_a-part_of_ont-BP.csv",
    "CC": "gene_go_matrix_propT_rel-is_a-part_of_ont-CC.csv",
    "MF": "gene_go_matrix_propT_rel-is_a-part_of_ont-MF.csv",
    "HPO": "gene_hpo_matrix_binary_withAncestors_namespace_Phenotypic_abnormality.csv", 
    "DepthBP": "goterm_depth_propT_rel-is_a-part_of_ont-BP.csv",
    "DepthCC": "goterm_depth_propT_rel-is_a-part_of_ont-CC.csv",
    "DepthMF": "goterm_depth_propT_rel-is_a-part_of_ont-MF.csv"
}

class DataLoader:
    def __init__(self, base_path: str = "../data/raw/", paths: dict = paths):
        self.base_path = base_path
        self.paths = paths

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Carica i dati dai file CSV e restituisce i rispettivi DataFrame.  
        """

        if not os.path.exists(self.base_path):
            raise FileNotFoundError(f"Base path '{self.base_path}' does not exist.")

        print(f"Loading DATA from {self.base_path}")

        try:
            BP_df = pd.read_csv(os.path.join(self.base_path, self.paths["BP"]), index_col=0)
            CC_df = pd.read_csv(os.path.join(self.base_path, self.paths["CC"]), index_col=0)
            MF_df = pd.read_csv(os.path.join(self.base_path, self.paths["MF"]), index_col=0)
            HPO_df = pd.read_csv(os.path.join(self.base_path, self.paths["HPO"]), index_col=0)
            DepthBP_df = pd.read_csv(os.path.join(self.base_path, self.paths["DepthBP"]), index_col=0)
            DepthCC_df = pd.read_csv(os.path.join(self.base_path, self.paths["DepthCC"]), index_col=0)
            DepthMF_df = pd.read_csv(os.path.join(self.base_path, self.paths["DepthMF"]), index_col=0)

            print(f"[OK] BP data loaded: {BP_df.shape}")
            print(f"[OK] CC data loaded: {CC_df.shape}")
            print(f"[OK] MF data loaded: {MF_df.shape}")
            print(f"[OK] HPO data loaded: {HPO_df.shape}")
            print(f"[OK] DepthBP data loaded: {DepthBP_df.shape}")
            print(f"[OK] DepthCC data loaded: {DepthCC_df.shape}")
            print(f"[OK] DepthMF data loaded: {DepthMF_df.shape}")

            return BP_df, CC_df, MF_df, HPO_df, DepthBP_df, DepthCC_df, DepthMF_df
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            raise

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise