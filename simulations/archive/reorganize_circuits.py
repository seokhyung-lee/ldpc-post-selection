import os
import re
import shutil


def reorganize_circuit_files():
    """
    Reorganize existing circuit files from old naming convention to new directory structure.

    Old format: (3,4)_hgp_circuit_n225_k9_d4_T12_p0.001_seed0.stim
    New format: (3,4)_n225_k9_d4_T12_p0.001/seed0.stim
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    circuit_dir = os.path.join(current_dir, "data/hgp_circuits")

    if not os.path.exists(circuit_dir):
        print(f"Circuit directory does not exist: {circuit_dir}")
        return

    # Pattern to match existing filenames
    pattern = r"\((\d+),(\d+)\)_hgp_circuit_n(\d+)_k(\d+)_d(\d+)_T(\d+)_p([\d.]+)_seed(\d+)\.stim"

    moved_files = 0
    errors = 0

    for filename in os.listdir(circuit_dir):
        if filename.endswith(".stim"):
            match = re.match(pattern, filename)
            if match:
                dv, dc, n, k, d, T, p, seed = match.groups()

                # Create new folder name (without "hgp_circuit_")
                folder_name = f"({dv},{dc})_n{n}_k{k}_d{d}_T{T}_p{p}"
                new_folder_path = os.path.join(circuit_dir, folder_name)

                # Create directory if it doesn't exist
                os.makedirs(new_folder_path, exist_ok=True)

                # New filename
                new_filename = f"seed{seed}.stim"

                # Source and destination paths
                src_path = os.path.join(circuit_dir, filename)
                dst_path = os.path.join(new_folder_path, new_filename)

                try:
                    # Move the file
                    shutil.move(src_path, dst_path)
                    print(f"Moved: {filename} -> {folder_name}/{new_filename}")
                    moved_files += 1
                except Exception as e:
                    print(f"Error moving {filename}: {e}")
                    errors += 1
            else:
                print(f"Filename doesn't match pattern: {filename}")

    print(f"\nReorganization complete!")
    print(f"Files moved: {moved_files}")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    reorganize_circuit_files()
