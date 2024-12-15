import CoreML
import Vision
import UIKit

class DrawingPredictor: ObservableObject {
    private var model: VNCoreMLModel?
    private var completion: ((String?) -> Void)?
    
    init() {
        do {
            // Load the Core ML model
            let config = MLModelConfiguration()
            let cnnModel = try HTR_CNN(configuration: config)
            model = try VNCoreMLModel(for: cnnModel.model)
        } catch {
            print("Error loading model: \(error)")
        }
    }
    
    func predict(drawing: UIImage, completion: @escaping (String?) -> Void) {
        self.completion = completion
        
        guard let model = model,
              let cgImage = drawing.cgImage else {
            completion(nil)
            return
        }
        
        // Create a Vision request with our Core ML model
        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            guard let results = request.results as? [VNClassificationObservation],
                  let topResult = results.first else {
                completion(nil)
                return
            }
            
            // Convert the predicted class index to a character
            let predictedClass = topResult.identifier
            DispatchQueue.main.async {
                self?.completion?(predictedClass)
            }
        }
        
        // Configure the image processing
        request.imageCropAndScaleOption = .centerCrop
        
        // Create an image handler and perform the request
        let handler = VNImageRequestHandler(cgImage: cgImage)
        do {
            try handler.perform([request])
        } catch {
            print("Failed to perform prediction: \(error)")
            completion(nil)
        }
    }
    
    func preprocessDrawing(_ drawing: UIImage) -> UIImage {
        // Convert the drawing to grayscale and resize to 28x28
        let size = CGSize(width: 28, height: 28)
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
        drawing.draw(in: CGRect(origin: .zero, size: size))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return resizedImage ?? drawing
    }
}
