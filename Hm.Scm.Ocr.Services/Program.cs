using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.AspNetCore.Mvc;
using OpenCvSharp;
using Sdcb.PaddleInference;
using Sdcb.PaddleOCR;
using Sdcb.PaddleOCR.Models;
using Sdcb.PaddleOCR.Models.Local;
using System.Diagnostics;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddSingleton(s =>
{
    Action<PaddleConfig> device = builder.Configuration["PaddleDevice"] == "GPU" ? PaddleDevice.Gpu() : PaddleDevice.Mkldnn();
    return new QueuedPaddleOcrAll(() => new PaddleOcrAll(LocalFullModels.ChineseV3, device)
    {
        Enable180Classification = true,
        AllowRotateDetection = true,
    }, consumerCount: 1);
});
builder.Services.Configure<FormOptions>((x) => 
{
    x.ValueLengthLimit = int.MaxValue;
    x.MultipartBoundaryLengthLimit = int.MaxValue;
});
builder.Services.AddHttpClient();
var app = builder.Build();
app.UseSwagger();
app.UseSwaggerUI();

app.MapPost("/appImageText", async (HttpContext context, string path) => 
{
    var httpClient = context.RequestServices.GetRequiredService<HttpClient>();
    using var stream =  await httpClient.GetStreamAsync(path);
    
    var queuedPaddleOcrAll =  app.Services.GetRequiredService<QueuedPaddleOcrAll>();
    using MemoryStream ms = new();
    stream.CopyTo(ms);


    using Mat src = Cv2.ImDecode(ms.ToArray(), ImreadModes.Color);
    double scale = 1;
    using Mat scaled = src.Resize(Size.Zero, scale, scale);

    Stopwatch sw = Stopwatch.StartNew();
    string textResult = ( queuedPaddleOcrAll.Run(scaled)).GetAwaiter().GetResult().Text;
    sw.Stop();
    Console.WriteLine(sw.ElapsedMilliseconds);
    return textResult;
}).WithName("GetAppImageText");
app.Run();
