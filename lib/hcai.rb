require "hcai/version"
require "pry"
require "matrix"

module Hcai
  class Error < StandardError; end

  class Analyzer
    SIGMOID = ->(e) { 1 / (1 + Math.exp(-e)) }
    TRAINER = 50_000

    attr_reader :synaptic_weights

    def initialize(synaptic_weights)
      @synaptic_weights = synaptic_weights
    end

    def train(input_data, result_data)
      TRAINER.times do
        output = (input_data * synaptic_weights).collect(&SIGMOID)

        @synaptic_weights += input_data.transpose() *
          (result_data - output).hadamard_product(output).hadamard_product(output.collect {|e| 1 - e})
      end
    end

    def think(data)
      (data * synaptic_weights).collect(&SIGMOID)
    end
  end

  def self.go
    analyzer = Hcai::Analyzer.new(Matrix.build(3, 1) { 0 })
    analyzer.train(
      Matrix[[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]],
      Matrix.column_vector([0, 1, 1, 0])
    )
    analyzer.predict(Matrix.row_vector([1, 0, 0]))
  end
end
