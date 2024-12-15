import SwiftUI
import PencilKit

struct ContentView: View {
    @State private var canvasView = PKCanvasView()
    @State private var predictedText: String = ""
    @State private var isLoading: Bool = false
    private let predictor = DrawingPredictor()
    
    var body: some View {
        VStack {
            Text("Draw a character")
                .font(.title)
                .padding()
            
            // Canvas for drawing
            CanvasView(canvasView: $canvasView)
                .frame(height: 300)
                .border(Color.gray)
            
            // Prediction result
            Text("Predicted: \(predictedText)")
                .font(.title)
                .padding()
            
            // Control buttons
            HStack {
                Button("Predict") {
                    predict()
                }
                .disabled(isLoading)
                
                Button("Clear") {
                    canvasView.drawing = PKDrawing()
                    predictedText = ""
                }
                .disabled(isLoading)
            }
            .padding()
        }
        .padding()
    }
    
    private func predict() {
        isLoading = true
        
        // Convert PKCanvasView to UIImage
        let image = canvasView.asImage()
        
        // Make prediction
        predictor.predict(drawing: image) { result in
            DispatchQueue.main.async {
                if let prediction = result {
                    predictedText = prediction
                } else {
                    predictedText = "Error"
                }
                isLoading = false
            }
        }
    }
}

// Helper view to wrap PKCanvasView
struct CanvasView: UIViewRepresentable {
    @Binding var canvasView: PKCanvasView
    
    func makeUIView(context: Context) -> PKCanvasView {
        canvasView.tool = PKInkingTool(.pen, color: .black, width: 15)
        canvasView.backgroundColor = .white
        return canvasView
    }
    
    func updateUIView(_ uiView: PKCanvasView, context: Context) {}
}

// Extension to convert PKCanvasView to UIImage
extension PKCanvasView {
    func asImage() -> UIImage {
        let renderer = UIGraphicsImageRenderer(bounds: bounds)
        let image = renderer.image { context in
            layer.render(in: context.cgContext)
        }
        
        // Resize to 28x28 and convert to grayscale
        let size = CGSize(width: 28, height: 28)
        UIGraphicsBeginImageContextWithOptions(size, false, 1)
        image.draw(in: CGRect(origin: .zero, size: size))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return resizedImage ?? image
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
