require "test_helper"

class HcaiTest < Minitest::Test
  def test_that_it_has_a_version_number
    refute_nil ::Hcai::VERSION
  end

  def test_it_predicts_result
    analyzer = Hcai::Analyzer.new(Matrix.build(3, 1) { 0 })
    analyzer.train(
      Matrix[
        [0, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
        [0, 1, 1]
      ],
      Matrix.column_vector(
        [
          0,
          1,
          1,
          0
        ]
      )
    )

    assert_equal(
      analyzer.think(
        Matrix.row_vector(
          [1, 0, 0]
        )
      ).first,
      0.9999370322638816
    )
  end

  def test_it_predicts_result
    analyzer = Hcai::Analyzer.new(Matrix.build(3, 1) { 0 })
    analyzer.train(
      Matrix[
        [0, 1, 1],
        [1, 0, 1],
        [0, 0, 1],
        [1, 1, 1],
      ],
      Matrix.column_vector(
        [
          1,
          0,
          0,
          1,
        ]
      )
    )

    assert_equal(
      analyzer.think(
        Matrix.row_vector(
          [1, 0, 1]
        )
      ).first,
      0.0
    )
  end
end
