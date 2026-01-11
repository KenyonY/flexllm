"""
Table processor for MLLM client
专门处理表格数据的处理器类
"""

import pandas as pd
from typing import List, Callable, Optional, Any, Union, TYPE_CHECKING

# 使用TYPE_CHECKING避免运行时循环引用
if TYPE_CHECKING:
    from .mllm_client import MllmClient


class MllmTableProcessor:
    """
    表格处理器类
    专门处理表格数据与MLLM客户端的交互
    """
    
    def __init__(self, mllm_client: "MllmClient"):
        """
        初始化表格处理器
        
        Args:
            mllm_client: MLLM客户端实例
        """
        self.mllm_client = mllm_client
    
    # 属性委托：委托给mllm_client的核心方法
    @property
    def call_llm(self):
        """委托给mllm_client的call_llm方法"""
        return self.mllm_client.call_llm
    
    @property
    def call_llm_with_selection(self):
        """委托给mllm_client的call_llm_with_selection方法"""
        return self.mllm_client.call_llm_with_selection
    
    @property
    def call_llm_sync(self):
        """委托给mllm_client的call_llm_sync方法"""
        return self.mllm_client.call_llm_sync
    
    # 数据预处理工具方法（独立于mllm_client）
    def process_text(self, text: str) -> str:
        """
        预处理文本
        
        Args:
            text: 输入文本
            
        Returns:
            str: 处理后的文本
        """
        return text
    
    def process_image(self, image: str) -> str:
        """
        预处理图像
        
        Args:
            image: 图片路径或URL
            
        Returns:
            str: 处理后的图片路径或URL
        """
        return image
    
    def load_dataframe(
        self,
        table_path: str,
        sheet_name: str = 0,
        max_num: int = None,
    ) -> pd.DataFrame:
        """
        加载并过滤Excel数据

        Args:
            table_path: 表格文件路径
            sheet_name: 要读取的sheet名称
            max_num: 最大处理数量限制

        Returns:
            处理后的DataFrame

        Raises:
            ValueError: 当输入参数无效时
            FileNotFoundError: 当文件不存在时
        """
        # 验证输入参数
        if not table_path:
            raise ValueError("表格文件路径不能为空")

        # 读取数据
        try:
            if table_path.endswith(".xlsx"):
                df = pd.read_excel(table_path, sheet_name=sheet_name)
            elif table_path.endswith(".csv"):
                df = pd.read_csv(table_path)
            else:
                raise ValueError(f"不支持的文件格式: {table_path}")
        except Exception as e:
            raise ValueError(f"读取文件失败: {str(e)}")

        if df.empty:
            print(f"警告: 过滤后数据为空")
            return df

        # 应用数量限制
        if max_num is not None:
            df = df.head(max_num)

        print(f"加载数据完成: {len(df)} 行")
        df = df.astype(str)
        print(f"{df.head(2)=}")

        return df
    
    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        image_col: Optional[str],
        text_col: str,
    ):
        """
        预处理df
        """
        df[text_col] = df[text_col].apply(self.process_text)
        # 只有当图像列存在时才处理图像列
        if image_col and image_col in df.columns:
            df[image_col] = df[image_col].apply(self.process_image)
        return df

    def _build_messages_from_dataframe(
        self,
        df: pd.DataFrame,
        image_col: Optional[str],
        text_col: str,
    ) -> List[List[dict]]:
        """
        从dataframe构建消息列表
        
        Args:
            df: 数据框
            image_col: 图片列名，如果为None或列不存在则只使用文本
            text_col: 文本列名
            
        Returns:
            messages_list: 消息列表
        """
        messages_list = []
        # 检查是否有图像列
        has_image_col = image_col and image_col in df.columns
        
        for index, row in df.iterrows():
            if has_image_col:
                # 有图像列时，包含图像和文本
                messages_list.append(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"{row[text_col]}"},
                                {"type": "image_url", "image_url": {"url": f"{str(row[image_col])}"}},
                            ],
                        }
                    ]
                )
            else:
                # 没有图像列时，只包含文本
                messages_list.append(
                    [
                        {
                            "role": "user", 
                            "content": f"{row[text_col]}"
                        }
                    ]
                )
        return messages_list

    def _build_image_messages_from_dataframe(
        self,
        df: pd.DataFrame,
        image_col: str,
        system_prompt: str = "",
        text_prompt: str = "请描述这幅图片",
    ) -> List[List[dict]]:
        """
        从dataframe构建图像处理消息列表
        
        Args:
            df: 数据框
            image_col: 图片列名
            system_prompt: 系统提示词
            text_prompt: 文本提示词
            
        Returns:
            messages_list: 消息列表
        """
        messages_list = []
        for index, row in df.iterrows():
            messages = []
            
            # 添加系统提示词（如果提供）
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            # 添加用户消息（包含文本提示和图像）
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {"type": "image_url", "image_url": {"url": f"{str(row[image_col])}"}},
                ],
            })
            
            messages_list.append(messages)
        return messages_list

    async def call_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str,
        image_col: Optional[str] = None,
        use_selection: bool = False,
        n_predictions: int = 3,
        selector_fn: Optional[Callable[[List[Any]], Any]] = None,
        **kwargs,
    ):
        """
        调用dataframe
        
        Args:
            df: 数据框
            image_col: 图片列名，如果为None或列不存在则只使用文本
            text_col: 文本列名
            use_selection: 是否使用选择模式
            n_predictions: 每条消息预测次数（仅在use_selection=True时有效）
            selector_fn: 选择函数（仅在use_selection=True时有效）
            **kwargs: 模型请求参数
            
        Returns:
            response_list: 响应列表
            
        Examples:
            # 纯文本模式（推荐的默认方式）
            await processor.call_dataframe(df, image_col=None, text_col="text")
            
            # 图像+文本模式（需要显式指定图像列）
            await processor.call_dataframe(df, image_col="image", text_col="text")
        """
        df = self.preprocess_dataframe(df, image_col, text_col)
        messages_list = self._build_messages_from_dataframe(df, image_col, text_col)
        
        if use_selection:
            return await self.call_llm_with_selection(
                messages_list, 
                n_predictions=n_predictions,
                selector_fn=selector_fn,
                **kwargs
            )
        else:
            return await self.call_llm(messages_list, **kwargs)

    
    async def call_table(
        self,
        table_path: str,
        text_col: str = "text",
        image_col: Optional[str] = None,
        sheet_name: str = 0,
        max_num: int = None,
        use_selection: bool = False,
        n_predictions: int = 1,
        selector_fn: Optional[Callable[[List[Any]], Any]] = None,
        **kwargs,
    ):
        """
        调用table
        
        Args:
            table_path: 表格文件路径
            image_col: 图片列名，默认为None（纯文本模式），指定列名则启用图像+文本模式
            text_col: 文本列名，默认为"text"
            sheet_name: sheet名称，默认为0
            max_num: 最大处理数量限制
            use_selection: 是否使用选择模式
            n_predictions: 每条消息预测次数（仅在use_selection=True时有效）
            selector_fn: 选择函数（仅在use_selection=True时有效）
            **kwargs: 其他参数
            
        Returns:
            response_list: 响应列表
        """
        df = self.load_dataframe(table_path, sheet_name, max_num)
        return await self.call_dataframe(
            df=df,
            text_col=text_col,
            image_col=image_col,
            use_selection=use_selection,
            n_predictions=n_predictions,
            selector_fn=selector_fn,
            **kwargs
        )
    

    async def call_table_images(
        self,
        table_path: str,
        image_col: str = "image",
        system_prompt: str = "",
        text_prompt: str = "请描述这幅图片",
        sheet_name: str = 0,
        max_num: int = None,
        use_selection: bool = False,
        n_predictions: int = 1,
        selector_fn: Optional[Callable[[List[Any]], Any]] = None,
        **kwargs,
    ):
        """
        对表格中的图像列进行批量请求大模型
        
        Args:
            table_path: 表格文件路径
            image_col: 图片列名，默认为"image"（此方法专门处理图像，必须指定有效的图像列）
            system_prompt: 系统提示词，默认为空
            text_prompt: 文本提示词，默认为"请描述这幅图片"
            sheet_name: sheet名称，默认为0
            max_num: 最大处理数量限制
            use_selection: 是否使用选择模式
            n_predictions: 每条消息预测次数（仅在use_selection=True时有效）
            selector_fn: 选择函数（仅在use_selection=True时有效）
            **kwargs: 其他参数
            
        Returns:
            response_list: 响应列表
        """
        df = self.load_dataframe(table_path, sheet_name, max_num)
        
        # 检查图像列是否存在
        if not image_col or image_col not in df.columns:
            raise ValueError(f"图像列 '{image_col}' 不存在于表格中")
        
        # 预处理图像列
        df[image_col] = df[image_col].apply(self.process_image)
        
        messages_list = self._build_image_messages_from_dataframe(
            df, image_col, system_prompt, text_prompt
        )
        
        if use_selection:
            return await self.call_llm_with_selection(
                messages_list,
                n_predictions=n_predictions,
                selector_fn=selector_fn,
                **kwargs
            )
        else:
            return await self.call_llm(messages_list, **kwargs)
