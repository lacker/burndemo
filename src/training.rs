use burn::nn::loss::CrossEntropyLossConfig;
use burn::prelude::Backend;
use burn::tensor::{Int, Tensor};
use burn::train::ClassificationOutput;

use crate::model::Model;

impl<B: Backend> Model<B> {
    pub fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}
