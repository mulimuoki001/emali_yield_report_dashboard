import pandas as pd
import tifffile

emali_data = tifffile.imread("data/lowlowlowres.tif")
channel_data = emali_data[:, :, 0]
df = pd.DataFrame(channel_data)
print(emali_data[:50, :50])
print(df.head)
