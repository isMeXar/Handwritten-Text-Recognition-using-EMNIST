import XCTest
@testable import HandwritingRecognition

class HandwritingRecognitionTests: XCTestCase {
    var predictor: DrawingPredictor!
    
    override func setUpWithError() throws {
        predictor = DrawingPredictor()
    }
    
    override func tearDownWithError() throws {
        predictor = nil
    }
    
    func testModelSelection() throws {
        // Test model switching
        predictor.setCurrentModel("CNN")
        predictor.setCurrentModel("KNN")
        predictor.setCurrentModel("NaiveBayes")
        // If no crash, test passes
    }
    
    func testPreprocessing() throws {
        // Create a test image
        let size = CGSize(width: 100, height: 100)
        UIGraphicsBeginImageContext(size)
        let context = UIGraphicsGetCurrentContext()
        context?.setFillColor(UIColor.white.cgColor)
        context?.fill(CGRect(origin: .zero, size: size))
        let testImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        
        // Test preprocessing
        let processedImage = predictor.preprocessDrawing(testImage)
        XCTAssertEqual(processedImage.size.width, 28)
        XCTAssertEqual(processedImage.size.height, 28)
    }
}
