import os

def list_complete_structure(output_filename="inria_ayrintili_rapor.txt"):
    """
    1. Mevcut calisma dizinini,
    2. Mevcut dizindeki (.py, .ipynb) dosyalarini,
    3. Alt klasörleri ve iceriklerini hiyerarsik olarak listeler.
    """
    current_path = os.path.abspath(os.getcwd())
    
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("--- AKADEMIK PROJE VE DOSYA ANALIZI ---\n\n")

            # 1. MEVCUT CALISMA DIZINI
            f.write(f"1. MEVCUT CALISMA DIZINI:\n")
            f.write(f"   > {current_path}\n\n")

            # 2. MEVCUT DIZINDEKI KOD DOSYALARI
            f.write("2. ANA DIZINDEKI KOD DOSYALARI:\n")
            with os.scandir(current_path) as entries:
                main_files = sorted([e.name for e in entries if e.is_file() and (e.name.endswith('.py') or e.name.endswith('.ipynb'))])
                
                if main_files:
                    for file in main_files:
                        f.write(f"   |-- [Kod] {file}\n")
                else:
                    f.write("   (Ana dizinde .py veya .ipynb dosyasi bulunamadi.)\n")
            
            f.write("\n")

            # 3. ALT KLASORLER VE ICERIKLERI
            f.write("3. ALT KLASORLER VE ICERIKLERI:\n")
            with os.scandir(current_path) as entries:
                subdirs = sorted([e for e in entries if e.is_dir()], key=lambda x: x.name)
                
                for subdir in subdirs:
                    f.write(f"\n   [Alt Klasör] {subdir.name}\n")
                    
                    try:
                        with os.scandir(subdir.path) as sub_entries:
                            items = sorted(list(sub_entries), key=lambda x: x.name)
                            
                            found_any = False
                            for item in items:
                                if item.is_dir():
                                    f.write(f"         |-- [Alt Dizin] {item.name}\n")
                                    found_any = True
                                elif item.is_file() and (item.name.endswith('.py') or item.name.endswith('.ipynb')):
                                    f.write(f"         |-- [Kod Dosyasi] {item.name}\n")
                                    found_any = True
                            
                            if not found_any:
                                f.write("         |-- (Ilgili icerik bulunamadi)\n")
                                
                    except PermissionError:
                        f.write("         |-- [Erisim Engellendi]\n")

            f.write("\n\n--- Rapor Sonu ---")
        
        print(f"Rapor basariyla olusturuldu: '{output_filename}'")

    except IOError as e:
        print(f"Dosya yazma hatasi: {e}")

if __name__ == "__main__":
    list_complete_structure()