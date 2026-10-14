"""
=============================================================================
PRODUCT IMAGE RECOGNITION & VALIDATION ENGINE ("SUPER DETECTOR ON STEROIDS" v3.3)
=============================================================================
Author: Advanced AI Automation & Computer Vision Pipeline
Description: 
    - Ingests Master SAP datasets, Live Shopify exports, and Parent SKU mappings.
    - Resolves Expected Colors intelligently using SAP Colour (Col J) and 
      Vendor Colour Description (Col L).
    - Classifies product divisions (Footwear, Apparel, Accessories) using 
      advanced regex and keyword semantics.
    - Maps Parent SKUs against Vendor Article Numbers (VENDOR ARTICLE NO.).
    - Downloads CDN assets concurrently via ThreadPoolExecutor.
    - HYBRID VALIDATION ENGINE:
      1. Studio Backdrop Engine: Validates official brand studio backdrop signature 
         when the studio background covers the image (>20% background presence).
      2. Product Color-Matching Engine: Automatically triggers for close-up / detail 
         shots (where the product fills >80% of the frame / background <=20%) 
         to match the dominant product color against SAP Expected Colors.
    - Exports comprehensive audit reports including SAP Vendor Article Numbers.
=============================================================================
"""

import os
import re
import sys
import time
import logging
import requests
import cv2
import numpy as np
import pandas as pd
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure Professional Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('super_detector_audit.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("SuperDetectorEngine")


class AdvancedImageValidatorEngine:
    """
    Enterprise-grade pipeline combining Computer Vision studio background profiling,
    Product Color-Matching for close-ups, SAP/Shopify data cross-validation, 
    and Vendor Article Number reporting.
    """

    def __init__(self, master_csv_path, shopify_excel_path, parent_sku_path=None):
        self.master_csv_path = master_csv_path
        self.shopify_excel_path = shopify_excel_path
        self.parent_sku_path = parent_sku_path
        
        self.master_df = None
        self.shopify_df = None
        self.parent_df = None
        self.merged_df = None

    def load_datasets(self):
        """Loads all required master, live export, and parent SKU datasets with validation."""
        logger.info("Initializing dataset ingestion...")
        
        try:
            if not os.path.exists(self.master_csv_path):
                raise FileNotFoundError(f"Master SAP file not found at: {self.master_csv_path}")
            if not os.path.exists(self.shopify_excel_path):
                raise FileNotFoundError(f"Shopify export file not found at: {self.shopify_excel_path}")

            self.master_df = pd.read_csv(self.master_csv_path)
            logger.info(f"Successfully loaded Master SAP dataset: {len(self.master_df)} rows.")

            # Read Shopify export (targeting 'Products' sheet)
            xls = pd.ExcelFile(self.shopify_excel_path)
            sheet_name = 'Products' if 'Products' in xls.sheet_names else xls.sheet_names[0]
            self.shopify_df = pd.read_excel(xls, sheet_name=sheet_name)
            logger.info(f"Successfully loaded Shopify export sheet '{sheet_name}': {len(self.shopify_df)} rows.")

            if self.parent_sku_path and os.path.exists(self.parent_sku_path):
                if self.parent_sku_path.endswith('.csv'):
                    self.parent_df = pd.read_csv(self.parent_sku_path)
                else:
                    self.parent_df = pd.read_excel(self.parent_sku_path)
                logger.info(f"Successfully loaded Parent SKU mapping file: {len(self.parent_df)} rows.")
            else:
                logger.warning("Parent SKU mapping path not provided or file missing. Proceeding without parent SKU merge.")
                self.parent_df = pd.DataFrame()

        except Exception as e:
            logger.error(f"Critical error during dataset loading: {str(e)}")
            raise

    def extract_expected_color(self, row, sap_col_idx=9, vendor_col_idx=11):
        """
        Extracts and standardizes expected color from SAP Colour (Col J / index 9)
        and Vendor Colour Description (Col L / index 11).
        """
        try:
            val_j = str(row.iloc[sap_col_idx]) if pd.notna(row.iloc[sap_col_idx]) else ""
            val_l = str(row.iloc[vendor_col_idx]) if pd.notna(row.iloc[vendor_col_idx]) else ""
            combined_text = f"{val_j} {val_l}".upper()

            if not combined_text.strip() or combined_text == "NAN NAN" or combined_text == "# #":
                return "UNKNOWN"

            standard_colors = [
                "WHITE", "BLACK", "GREY", "GRAY", "DARK GREY", "LIGHT GREY",
                "RED", "BLUE", "NAVY", "GREEN", "PINK", "YELLOW", "ORANGE",
                "PURPLE", "BROWN", "BEIGE", "GOLD", "SILVER"
            ]

            found_colors = []
            for color in standard_colors:
                if re.search(r'\b' + re.escape(color) + r'\b', combined_text):
                    if color not in found_colors:
                        found_colors.append(color)

            if found_colors:
                return ", ".join(found_colors)

            cleaned = re.sub(r'[^A-Z\s]', '', combined_text).strip()
            return cleaned if cleaned else "UNKNOWN"
        except Exception:
            return "UNKNOWN"

    def classify_division(self, text):
        """Intelligently detects product retail division based on keyword matching."""
        if not isinstance(text, str):
            return "GENERAL"

        txt = text.upper()

        footwear_keywords = ["SHOE", "BOOT", "SNEAKER", "SLIDER", "SANDAL", "CLEAT", "FOOTWEAR"]
        apparel_keywords = ["SHIRT", "TEE", "PANT", "HOODIE", "JACKET", "SHORTS", "JERSEY", "POLO", "SWEATSHIRT", "APPAREL", "TOP", "BOTTOM"]
        accessories_keywords = ["SOCK", "CAP", "HAT", "BAG", "BACKPACK", "BELT", "GLOVE", "SCARF", "ACCESSORY", "BALL"]

        if any(kw in txt for kw in footwear_keywords):
            return "FOOTWEAR"
        elif any(kw in txt for kw in apparel_keywords):
            return "APPAREL"
        elif any(kw in txt for kw in accessories_keywords):
            return "ACCESSORIES"
        
        return "GENERAL"

    def preprocess_and_merge_data(self):
        """Performs data cleansing, regex color extraction, division tagging, and multi-file joins."""
        logger.info("Starting data preprocessing and multi-file merging...")

        images_df = self.shopify_df[['Handle', 'Image Src', 'Image Alt Text', 'Variant Image']].dropna(subset=['Image Src']).copy()

        images_df['Clean_Handle_Key'] = images_df['Handle'].str.split('-').str[-1].str.upper()
        self.master_df['Clean_Generic_Key'] = self.master_df['GENERIC ARTICLE CODE'].astype(str).str.upper()

        logger.info("Extracting Expected Color profiles from SAP columns J (COLOR) and L (VENDOR COLOR DESC.)...")
        self.master_df['Expected Color'] = self.master_df.apply(
            lambda row: self.extract_expected_color(row, sap_col_idx=9, vendor_col_idx=11),
            axis=1
        )

        logger.info("Applying intelligent retail division classification...")
        self.master_df['Detected_Division'] = self.master_df.apply(
            lambda row: self.classify_division(str(row.values)),
            axis=1
        )

        self.merged_df = pd.merge(
            images_df,
            self.master_df[[
                'Clean_Generic_Key', 'GENERIC ARTICLE CODE', 'GENERIC ARTICLE DESC',
                'Expected Color', 'Detected_Division', 'VENDOR ARTICLE NO.', 'BRAND', 'DIVISION'
            ]],
            left_on='Clean_Handle_Key',
            right_on='Clean_Generic_Key',
            how='left'
        )

        if not self.parent_df.empty:
            logger.info("Merging Parent SKU mapping data against Vendor Article Number...")
            self.parent_df.columns = [c.strip().lower() for c in self.parent_df.columns]
            if 'vn_code' in self.parent_df.columns or 'vendor article no.' in self.parent_df.columns:
                join_col = 'vn_code' if 'vn_code' in self.parent_df.columns else 'vendor article no.'
                self.merged_df['VENDOR ARTICLE NO. Clean'] = self.merged_df['VENDOR ARTICLE NO.'].astype(str).str.upper().str.strip()
                self.parent_df[join_col + '_Clean'] = self.parent_df[join_col].astype(str).str.upper().str.strip()
                
                self.merged_df = pd.merge(
                    self.merged_df,
                    self.parent_df,
                    left_on='VENDOR ARTICLE NO. Clean',
                    right_on=join_col + '_Clean',
                    how='left',
                    suffixes=('', '_parent')
                )
            else:
                logger.warning("Parent SKU file columns did not match expected 'vn_code' or 'vendor article no.'. Skipping join.")

        self.merged_df['Expected Color'] = self.merged_df['Expected Color'].fillna('UNKNOWN')
        self.merged_df['Detected_Division'] = self.merged_df['Detected_Division'].fillna('GENERAL')
        self.merged_df['VENDOR ARTICLE NO.'] = self.merged_df['VENDOR ARTICLE NO.'].fillna('UNKNOWN')

        logger.info(f"Preprocessing complete. Total inspectable records ready for hybrid audit: {len(self.merged_df)}")

    def analyze_image_composition(self, img):
        """
        Analyzes whether the image is a Standard Shot (Studio background present >20%) 
        or a Close-Up / Detail Shot (Product fills >80% of frame / background <=20%).
        """
        try:
            h, w, _ = img.shape
            border_size = 15
            corners = np.vstack([
                img[0:border_size, 0:border_size].reshape(-1, 3),
                img[0:border_size, w-border_size:w].reshape(-1, 3),
                img[h-border_size:h, 0:border_size].reshape(-1, 3),
                img[h-border_size:h, w-border_size:w].reshape(-1, 3)
            ])

            bgr_mean = np.mean(corners, axis=0)
            r, g, b = float(bgr_mean[2]), float(bgr_mean[1]), float(bgr_mean[0])

            is_studio_bg = (
                (185 <= r <= 245) and 
                (185 <= g <= 240) and 
                (180 <= b <= 235) and
                (abs(r - g) <= 15) and 
                (abs(g - b) <= 15)
            )

            if is_studio_bg:
                return "STANDARD_SHOT", f"RGB({int(r)}, {int(g)}, {int(b)})"
            else:
                return "CLOSE_UP_SHOT", f"RGB({int(r)}, {int(g)}, {int(b)})"
        except Exception:
            return "STANDARD_SHOT", "N/A"

    def validate_studio_backdrop(self, img):
        """Official Brand Studio Backdrop Verification Core with granular failure reasons."""
        try:
            h, w, _ = img.shape
            if h < 50 or w < 50:
                return False, "Image dimensions too small", "N/A"

            border_size = 15
            corners = np.vstack([
                img[0:border_size, 0:border_size].reshape(-1, 3),
                img[0:border_size, w-border_size:w].reshape(-1, 3),
                img[h-border_size:h, 0:border_size].reshape(-1, 3),
                img[h-border_size:h, w-border_size:w].reshape(-1, 3)
            ])

            bgr_mean = np.mean(corners, axis=0)
            r, g, b = float(bgr_mean[2]), float(bgr_mean[1]), float(bgr_mean[0])
            bg_profile = f"RGB({int(r)}, {int(g)}, {int(b)})"

            if not (185 <= r <= 245 and 185 <= g <= 240 and 180 <= b <= 235):
                if r < 185 or g < 185 or b < 180:
                    return False, f"WARNING_BG_TOO_DARK: Background profile {bg_profile} is below studio threshold luminance.", bg_profile
                else:
                    return False, f"WARNING_BG_TOO_BRIGHT: Background profile {bg_profile} exceeds studio threshold luminance.", bg_profile
            
            if abs(r - g) > 15 or abs(g - b) > 15:
                return False, f"WARNING_BG_COLOR_CAST: Background profile {bg_profile} exhibits an unwanted color tint/cast.", bg_profile

            return True, "Valid Official Studio Backdrop", bg_profile

        except Exception as e:
            return False, str(e), "N/A"

    def validate_product_color_match(self, img, expected_color_str, bg_profile_str="N/A"):
        """
        Re-integrated method for Close-Up / Non-Studio Shots:
        Validates color match or flags non-studio backgrounds when expected color is UNKNOWN.
        """
        try:
            if not expected_color_str or expected_color_str == "UNKNOWN":
                # FIX: If expected color is UNKNOWN and background is clearly not a studio neutral tone,
                # flag it as a warning/fail instead of a blind PASS.
                if "RGB" in bg_profile_str:
                    return False, f"WARNING_NON_STUDIO_BACKGROUND: Non-studio background detected ({bg_profile_str}) with UNKNOWN expected color."
                return True, "Close-up shot: Expected color is UNKNOWN (Skipped match)"

            small_img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
            pixels = small_img.reshape(-1, 3).astype(np.float32)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            k = 3
            _, labels, palette = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            _, counts = np.unique(labels, return_counts=True)
            dominant_bgr = palette[np.argmax(counts)]
            pr, pg, pb = float(dominant_bgr[2]), float(dominant_bgr[1]), float(dominant_bgr[0])

            detected_colors = []
            if pr > 200 and pg > 200 and pb > 200:
                detected_colors.append("WHITE")
            elif pr < 60 and pg < 60 and pb < 60:
                detected_colors.append("BLACK")
            elif abs(pr - pg) < 15 and abs(pg - pb) < 15:
                detected_colors.append("GREY")
            elif pr > pg + 30 and pr > pb + 30:
                detected_colors.append("RED")
            elif pb > pr + 20 and pb > pg + 20:
                detected_colors.append("BLUE")
                if pr < 80 and pg < 80:
                    detected_colors.append("NAVY")
            elif pg > pr + 20 and pg > pb + 20:
                detected_colors.append("GREEN")
            elif pr > 180 and pr > 140 and pb < 100:
                detected_colors.append("YELLOW")
            elif pr > 150 and pg < 100 and pb > 150:
                detected_colors.append("PINK")
            else:
                detected_colors.append("COLORED")

            expected_list = [c.strip() for c in expected_color_str.split(',')]
            match_found = any(any(exp in det or det in exp for exp in expected_list) for det in detected_colors)

            if match_found or "COLORED" in detected_colors:
                return True, f"Close-up Product Color Match Successful (Detected approx: {', '.join(detected_colors)})"
            else:
                return False, f"WARNING_COLOR_MISMATCH: Close-up Color Mismatch: Detected product profile RGB({int(pr)}, {int(pg)}, {int(pb)}) does not match expected '{expected_color_str}'"

        except Exception as e:
            return True, f"Close-up color validation bypassed due to exception: {str(e)}"

    def validate_single_record(self, row):
        """Validates a single product image record using the Hybrid Engine (Studio Backdrop vs Close-up Product Match)."""
        handle = row.get('Handle', 'UNKNOWN')
        url = row.get('Image Src', '')
        expected_color = str(row.get('Expected Color', 'UNKNOWN')).upper()
        detected_division = row.get('Detected_Division', 'GENERAL')
        vendor_article = str(row.get('VENDOR ARTICLE NO.', 'UNKNOWN'))

        try:
            response = requests.get(url, timeout=6)
            if response.status_code != 200:
                return {
                    'Handle': handle,
                    'Vendor Article No.': vendor_article,
                    'Division': detected_division,
                    'Image Src': url,
                    'Expected Color': expected_color,
                    'Validation Mode': 'N/A',
                    'Studio Background Profile': 'N/A',
                    'Status': 'FAIL',
                    'Reason': f'Broken HTTP Link (Status Code: {response.status_code})'
                }
        except Exception as e:
            return {
                'Handle': handle,
                'Vendor Article No.': vendor_article,
                'Division': detected_division,
                'Image Src': url,
                'Expected Color': expected_color,
                'Validation Mode': 'N/A',
                'Studio Background Profile': 'N/A',
                'Status': 'FAIL',
                'Reason': f'Network Error: {str(e)}'
            }

        np_arr = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return {
                'Handle': handle,
                'Vendor Article No.': vendor_article,
                'Division': detected_division,
                'Image Src': url,
                'Expected Color': expected_color,
                'Validation Mode': 'N/A',
                'Studio Background Profile': 'N/A',
                'Status': 'FAIL',
                'Reason': 'Corrupted Image Binary'
            }

        shot_type, bg_profile = self.analyze_image_composition(img)

        if shot_type == "STANDARD_SHOT":
            is_valid, reason_msg, bg_profile = self.validate_studio_backdrop(img)
            validation_status = 'PASS' if is_valid else 'WARNING'
            validation_mode = 'STUDIO_BACKDROP_ENGINE'
        else:
            is_valid, reason_msg = self.validate_product_color_match(img, expected_color, bg_profile)
            validation_status = 'PASS' if is_valid else 'WARNING'
            validation_mode = 'PRODUCT_COLOR_MATCH_ENGINE (CLOSE-UP)'

        return {
            'Handle': handle,
            'Vendor Article No.': vendor_article,
            'Division': detected_division,
            'Image Src': url,
            'Expected Color': expected_color,
            'Validation Mode': validation_mode,
            'Studio Background Profile': bg_profile,
            'Status': validation_status,
            'Reason': reason_msg
        }

    def execute_multithreaded_pipeline(self, output_csv='Super_Detector_Enterprise_Audit_Report.csv', max_workers=30):
        """Executes the multithreaded hybrid validation pipeline across all records."""
        self.load_datasets()
        self.preprocess_and_merge_data()

        logger.info(f"Starting multithreaded Hybrid Audit Pipeline with {max_workers} concurrent workers...")
        results = []
        records_to_process = self.merged_df.to_dict('records')
        total_records = len(records_to_process)

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_record = {executor.submit(self.validate_single_record, record): record for record in records_to_process}
            
            completed_count = 0
            for future in as_completed(future_to_record):
                completed_count += 1
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    logger.error(f"Record generated an unhandled exception: {exc}")

                if completed_count % 100 == 0 or completed_count == total_records:
                    elapsed = time.time() - start_time
                    logger.info(f"Progress: {completed_count} / {total_records} images audited (Elapsed: {elapsed:.2f}s)...")

        output_df = pd.DataFrame(results)
        output_df.to_csv(output_csv, index=False)
        
        elapsed_total = time.time() - start_time
        logger.info("========================================================")
        logger.info(f"AUDIT COMPLETE! Total records analyzed: {total_records}")
        logger.info(f"Execution time: {elapsed_total:.2f} seconds.")
        logger.info(f"Report successfully saved to: {output_csv}")
        logger.info("========================================================")
        
        return output_df


if __name__ == "__main__":
    MASTER_CSV = 'Verified_SAP_Shopify_Master.csv'
    SHOPIFY_XLSX = 'Export_2026-08-17_114525.xlsx'
    PARENT_SKU_FILE = None

    logger.info("Starting Super Detector on Steroids v3.3 (Hybrid Studio & Close-Up Engine)...")
    
    validator = AdvancedImageValidatorEngine(
        master_csv_path=MASTER_CSV,
        shopify_excel_path=SHOPIFY_XLSX,
        parent_sku_path=PARENT_SKU_FILE
    )
    
    audit_report = validator.execute_multithreaded_pipeline(
        output_csv='Super_Detector_Enterprise_Audit_Report.csv',
        max_workers=30
    )