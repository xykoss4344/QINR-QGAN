import nbformat as nbf

nb = nbf.read('QGAN_Crystal_Analysis.ipynb', as_version=4)

new_markdown = nbf.v4.new_markdown_cell('## 4. Structural Coordinate Comparison (MSE & Euclidean Distance)\nInstead of just looking at the overarching Critic distribution, this cell calculates the actual geometric **Mean Squared Error (MSE)** and the **Average Euclidean Distance** between every Generated crystal and its mathematically closest Real crystal counterpart.')

new_code = nbf.v4.new_code_cell('''from scipy.spatial.distance import cdist

print("Calculating Structural Similarities by comparing 3D Coordinates...")

# 1. Flatten all coordinates to compare them mathematically
# real_coords shape: (1000, 90)
# fake_coords shape: (1000, 90)
real_flat = real_coords.cpu().numpy() 
fake_flat = fake_coords.cpu().numpy() 

# 2. Find the closest Real crystal for each Fake crystal
# cdist calculates the geometric distance between every fake crystal and every real crystal
all_distances = cdist(fake_flat, real_flat, metric='euclidean')

# Find the minimum distance for each fake crystal (its closest real neighbor)
min_distances = np.min(all_distances, axis=1)

# 3. Calculate Mean Squared Error (MSE) and Average Minimum Distance
mse_scores = min_distances ** 2 / 90 # Dividing by 90 coordinates
avg_mse = np.mean(mse_scores)
avg_euc_dist = np.mean(min_distances)

print("\\n===========================================")
print("     DIRECT STRUCTURAL COMPARISON RESULTS   ")
print("===========================================")
print(f"{str('Average MSE'):25} : {avg_mse:.5f}")
print(f"{str('Average Coordinate Dist'):25} : {avg_euc_dist:.5f}")
print(f"{str('Closest Single Match Dist'):25} : {np.min(min_distances):.5f}")
print("===========================================\\n")

print("WHAT DO THESE VALUES MEAN?\\n")
print("1. Average MSE (Mean Squared Error):")
print("   How much the coordinates of your generated crystal differ from the real crystal on average per coordinate axis.")
print("   A value of 0.0 means the AI copied a real crystal perfectly.")
print(f"   Current Value ({avg_mse:.5f}) means that on average, the atoms are slightly shifted away from their ideal real-world positions.\\n")

print("2. Average Coordinate Dist:")
print("   This is the total geometric Euclidean distance between the fake crystal and its nearest real counterpart.")
print(f"   Current Value ({avg_euc_dist:.5f}) represents the total magnitude of error accumulated across all 30 atoms.\\n")

print("3. Closest Single Match Dist:")
print("   Out of all 1,000 generated crystals, this tells us how close the absolute *best* fake crystal got to reality.")
print(f"   The absolute best QGAN-generated crystal had a total geometric distance of {np.min(min_distances):.5f} from a real crystal.")
''')

# Insert the new cells before the Histogram plotting cells (which are the last two cells in the current nb structure)
nb.cells.insert(7, new_markdown)
nb.cells.insert(8, new_code)

# Rename the final sections to maintain numbering
nb.cells[9].source = nb.cells[9].source.replace('## 4.', '## 5.')
if len(nb.cells) > 11:
    nb.cells[11].source = nb.cells[11].source.replace('## 5.', '## 6.')

nbf.write(nb, 'QGAN_Crystal_Analysis.ipynb')
print("Notebook updated successfully.")
