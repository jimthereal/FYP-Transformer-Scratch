import re

def fix_import_path():
    """Fix the import path in enhanced_preprocessing.py"""
    file_path = "src/enhanced_preprocessing.py"
    
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix the import path
        # Change: from models.lipnet_base import LipNet
        # To: from models.lipnet_base import LipNet
        old_import = "from models.lipnet_base import LipNet"
        new_import = "from models.lipnet_base import LipNet"
        
        if old_import in content:
            content = content.replace(old_import, new_import)
            
            # Write back the fixed content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("✅ Fixed import path in enhanced_preprocessing.py")
            print(f"   Changed: {old_import}")
            print(f"   To: {new_import}")
            return True
        else:
            print("❌ Import path not found or already fixed")
            return False
            
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return False
    except Exception as e:
        print(f"❌ Error fixing import: {e}")
        return False

def test_enhanced_preprocessing():
    """Test the enhanced preprocessing after fixing imports"""
    print("\n🧪 Testing enhanced preprocessing...")
    print("Run this command from your project root:")
    print("python src/enhanced_preprocessing.py")
    print("\nExpected output:")
    print("- 75/75 frames processed successfully")
    print("- Tensor shape: [1, 3, 75, 64, 128]")
    print("- Some prediction output (may still be gibberish until Phase 2B)")

if __name__ == "__main__":
    print("🔧 LipNet FYP - Quick Fix Script")
    print("=" * 50)
    
    success = fix_import_path()
    
    if success:
        test_enhanced_preprocessing()
    else:
        print("\n⚠️  Please check the file path and try again")