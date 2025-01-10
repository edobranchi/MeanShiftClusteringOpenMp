#include <chrono>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <omp.h>

using namespace cv;
using namespace std;

// distanza euclidea √((x₁-x₂)² + (y₁-y₂)² + (z₁-z₂)²)
inline float euclideanDistance(const Vec3f& a, const Vec3f& b) {
    return sqrt((a[0] - b[0]) * (a[0] - b[0]) +
                (a[1] - b[1]) * (a[1] - b[1]) +
                (a[2] - b[2]) * (a[2] - b[2]));
}


void meanShift(const Mat& input, Mat& output, float radius, int max_iter = 15, float epsilon = 1e-3) {
    //Conversione in floating point perchè sennò con gli interi impazzisco
    Mat data;
    input.convertTo(data, CV_32FC3);

    // Converto immagine in vettore dove ogni elemento contiene i colori RGB
    vector<Vec3f> points;
    for (int y = 0; y < data.rows; ++y) {
        for (int x = 0; x < data.cols; ++x) {
            points.push_back(data.at<Vec3f>(y, x));
        }
    }

    // alloco il vettore per i punti spostati
    vector<Vec3f> shiftedPoints(points.size());

    // eseguo mean-shift in parallelo, ciclo su ogni pixel e lo prendo come punto di partenza
    #pragma omp parallel for
    for (size_t i = 0; i < points.size(); ++i) {
        Vec3f currentPoint = points[i];
        for (int iter = 0; iter < max_iter; ++iter) {
            Vec3f newPoint = Vec3f(0, 0, 0);
            float totalWeight = 0;
            for (const auto& neighbor : points) {
                float distance = euclideanDistance(currentPoint, neighbor);
                if (distance < radius) {
                    float weight = (distance < radius) ? 1.0 : 0.0;
                    newPoint += weight * neighbor;
                    totalWeight += weight;
                }
            }
            newPoint /= totalWeight;
            if (euclideanDistance(currentPoint, newPoint) < epsilon) break;
            currentPoint = newPoint;
        }
        shiftedPoints[i] = currentPoint;
    }

    // ricostruisco l'immagine
    output = Mat(data.rows, data.cols, CV_32FC3);
    int index = 0;
    for (int y = 0; y < output.rows; ++y) {
        for (int x = 0; x < output.cols; ++x) {
            output.at<Vec3f>(y, x) = shiftedPoints[index++];
        }
    }

    // riporto l'immagine in 8 bit
    output.convertTo(output, CV_8UC3);
}

int main() {
    //carico l'immagine di input
    Mat input = imread("/home/edoardo/CLionProjects/MeanShiftClusterOpenMP/test_images/promuoviamo-paesaggio-italiano.jpg");
    Mat originalInput = imread("/home/edoardo/CLionProjects/MeanShiftClusterOpenMP/test_images/promuoviamo-paesaggio-italiano.jpg");
    resize(input, input, Size(), 0.5, 0.5, INTER_AREA);
    resize(originalInput, originalInput, Size(), 0.5, 0.5, INTER_AREA);

    //converto da BGR a RGB e da RGB a Lab
    cvtColor(input, input, COLOR_BGR2RGB);
    cvtColor(input, input, COLOR_RGB2Lab);

    Mat output;
    //raggio di ricerca dei vicini
    float radius = 5.0;

    //eseguo mean-shift e misuro tempo di esecuzione
    auto start = std::chrono::high_resolution_clock::now();
    meanShift(input, output, radius);
    auto end = std::chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "Tempo di esecuzione con OpenMP: " << elapsed.count() << " secondi" << endl;

    cvtColor(output, output, COLOR_Lab2RGB);
    cvtColor(output, output, COLOR_RGB2BGR);

    imshow("Processata", input);
    imshow("Originale", output);
    waitKey(0);

    return 0;
}
