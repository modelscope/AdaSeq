import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


class TestPipelines(unittest.TestCase):
    def test_span_based_ner_pipeline(self):
        pipeline_ins = pipeline(
            Tasks.named_entity_recognition,
            'damo/nlp_nested-ner_named-entity-recognition_chinese-base-med',
        )
        print(
            pipeline_ins(
                '1、可测量目标： 1周内胸闷缓解。2、下一步诊疗措施：1.心内科护理常规，一级护理，低盐低脂饮食，留陪客。2.予“阿司匹林肠溶片”抗血小板聚集，“呋塞米、螺内酯”利尿减轻心前负荷，“瑞舒伐他汀”调脂稳定斑块，“厄贝沙坦片片”降血压抗心机重构'
            )
        )


if __name__ == '__main__':
    unittest.main()
