
# predict test
python src/predict.py "models/data-processor.pickle" "models/hLSTMat-70" "${1}/testing_id.txt" "${1}/testing_data" "${2}" --arch hLSTMat

# predict peer review
python src/predict.py "models/data-processor.pickle" "models/hLSTMat-70" "${1}/peer_review_id.txt" "${1}/peer_review" "${3}" --arch hLSTMat
