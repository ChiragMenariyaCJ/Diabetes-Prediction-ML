import axios from "axios"

export const PredictDiabetesService = {
    getPredictionData: (inputData) => {
        const url = 'http://localhost:3000/'
        return axios.post(url, inputData)
    }
}