import os
from datetime import datetime
from PIL import Image
import pandas as pd
from pathlib import Path

def analyze_images(folder_path):
    image_data = []
    
    # Get all jpg files
    files = list(Path(folder_path).glob("*.jpg"))
    print(f"Found {len(files)} images")
    
    for file_path in files:
        try:
            # Get file info
            stats = file_path.stat()
            
            # Open image to get dimensions
            with Image.open(file_path) as img:
                width, height = img.size
            
            # Parse date from filename (format: 2014-11-26_18-22-20_UTC.jpg)
            date_str = file_path.stem.split('_')[0]  # Gets "2014-11-26"
            date = datetime.strptime(date_str, '%Y-%m-%d')
            
            # Collect data
            image_info = {
                'filename': file_path.name,
                'date': date,
                'year': date.year,
                'month': date.month,
                'size_bytes': stats.st_size,
                'size_mb': stats.st_size / (1024 * 1024),  # Convert to MB
                'width': width,
                'height': height,
                'aspect_ratio': width/height
            }
            
            image_data.append(image_info)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(image_data)
    
    # Sort by date
    df = df.sort_values('date')
    
    # Basic statistics
    print("\n=== Image Collection Statistics ===")
    print(f"Total images: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Average size: {df['size_mb'].mean():.2f} MB")
    print(f"Total size: {df['size_mb'].sum():.2f} MB")
    
    # Images per year
    print("\n=== Images by Year ===")
    year_counts = df['year'].value_counts().sort_index()
    print(year_counts)
    
    # Save detailed report
    report_path = 'image_analysis.csv'
    df.to_csv(report_path, index=False)
    print(f"\nDetailed report saved to {report_path}")
    
    # Return DataFrame for further analysis
    return df

if __name__ == "__main__":
    folder_path = "christmasfreud"
    print(f"Analyzing images in {folder_path}...")
    df = analyze_images(folder_path)
    
    # Show the first few entries
    print("\n=== Sample of Image Data ===")
    print(df.head())
    
    # Optional: Create year-based folders and sort images
    create_folders = input("\nWould you like to sort images into year folders? (y/n): ")
    if create_folders.lower() == 'y':
        for year in df['year'].unique():
            year_folder = os.path.join(folder_path, str(year))
            os.makedirs(year_folder, exist_ok=True)
            
            year_files = df[df['year'] == year]['filename']
            for filename in year_files:
                src = os.path.join(folder_path, filename)
                dst = os.path.join(year_folder, filename)
                os.rename(src, dst)
                print(f"Moved {filename} to {year_folder}")
