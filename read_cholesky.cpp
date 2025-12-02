// TL+ {"workspace_files": ["Cholesky_64x64.bin", "test_a_16x3072.bin"]}
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

int main() {
    std::cout << "Program started!" << std::endl;
    std::cout.flush();
    
    const int n = 64;  // 64x64 matrix
    const std::string filename = "test_a_16x3072.bin";
    
    std::cout << "Looking for file: " << filename << std::endl;
    
    // Open file in binary mode
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return 1;
    }
    
    // Read all floats
    std::vector<float> data(n * n);
    file.read(reinterpret_cast<char*>(data.data()), n * n * sizeof(float));
    
    // Check if read was successful
    if (file.fail()) {
        std::cerr << "Error: Failed to read file " << filename << std::endl;
        return 1;
    }
    
    std::cout << "Successfully read " << n << "x" << n << " matrix from " << filename << std::endl;
    std::cout << "Total elements: " << n * n << std::endl;
    std::cout << "File size: " << n * n * sizeof(float) << " bytes" << std::endl;
    
    // Print first few elements
    std::cout << "\nFirst 5x5 block:" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            std::cout << std::setw(10) << data[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Print diagonal elements
    std::cout << "\nDiagonal elements:" << std::endl;
    for (int i = 0; i < n; i += 8) {  // Print every 8th diagonal
        std::cout << "L[" << i << "," << i << "] = " << data[i * n + i] << std::endl;
    }
    
    // Print some statistics
    float min_val = data[0];
    float max_val = data[0];
    float sum = 0.0f;
    int zero_count = 0;
    
    for (float val : data) {
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
        if (val == 0.0f) zero_count++;
    }
    
    std::cout << "\nStatistics:" << std::endl;
    std::cout << "Min value: " << min_val << std::endl;
    std::cout << "Max value: " << max_val << std::endl;
    std::cout << "Mean value: " << sum / (n * n) << std::endl;
    std::cout << "Number of zeros: " << zero_count << " (" 
              << (100.0f * zero_count / (n * n)) << "%)" << std::endl;
    
    // Verify it's lower triangular (for Cholesky result)
    std::cout << "\nChecking if lower triangular..." << std::endl;
    bool is_lower_triangular = true;
    int upper_nonzero = 0;
    
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {  // Check upper triangle
            if (data[i * n + j] != 0.0f) {
                is_lower_triangular = false;
                upper_nonzero++;
            }
        }
    }
    
    if (is_lower_triangular) {
        std::cout << "✓ Matrix is lower triangular" << std::endl;
    } else {
        std::cout << "✗ Matrix is NOT lower triangular (" 
                  << upper_nonzero << " non-zero elements in upper triangle)" << std::endl;
    }
    
    file.close();
    return 0;
}