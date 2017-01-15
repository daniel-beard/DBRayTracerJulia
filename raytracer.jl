
#= Raytracer in Julia =#

#==============================================================
Vectors
==============================================================#

type Vec3
  x::Float64
  y::Float64
  z::Float64
end

Vec3Zero()::Vec3 = Vec3(0,0,0)
vec_sqrt(v::Vec3)::Vec3 = Vec3(sqrt(v.x), sqrt(v.y), sqrt(v.z))

function float_eq(a::Float64, b::Float64)::Bool
  abs(a - b) < 0.0001
end

function vec_eq(a::Vec3, b::Vec3)::Bool
  float_eq(a.x, b.x) && float_eq(a.y, b.y) && float_eq(a.z, b.z)
end

function dot(a::Vec3, b::Vec3)::Float64
  a.x*b.x + a.y*b.y + a.z*b.z
end

function cross(a::Vec3, b::Vec3)::Vec3
  return Vec3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, x*b.y-a.y*b.x)
end

function lengthSquared(a::Vec3)::Float64
  a.x*a.x + a.y*a.y + a.z*a.z
end

function length(a::Vec3)::Float64
  sqrt(lengthSquared(a))
end

function normalized(a::Vec3)::Vec3
  l_squared = lengthSquared(a)
  if float_eq(l_squared, 0) || float_eq(l_squared, 1)
    return a
  end
  return div(a, sqrt(l_squared))
end

function add(a::Vec3, b::Vec3)::Vec3
  Vec3(a.x+b.x, a.y+b.y, a.z+b.z)
end

function add(a::Vec3, b::Float64)::Vec3
  Vec3(a.x+b, a.y+b, a.z+b)
end

function subtract(a::Vec3, b::Vec3)::Vec3
  Vec3(a.x-b.x, a.y-b.y, a.z-b.z)
end

function mul(a::Vec3, b::Vec3)::Vec3
  Vec3(a.x*b.x, a.y*b.y, a.z*b.z)
end

function mul(a::Vec3, b::Float64)::Vec3
  Vec3(a.x*b, a.y*b, a.z*b)
end

function mul(a::Float64, b::Vec3)::Vec3
  Vec3(a*b.x, a*b.y, a*b.z)
end

function div(a::Vec3, b::Vec3)::Vec3
  Vec3(a.x/b.x, a.y/b.y, a.z/b.z)
end

function div(a::Vec3, b::Float64)::Vec3
  Vec3(a.x/b, a.y/b, a.z/b)
end

function eq(a::Vec3, b::Vec3)::Bool
  a.x==b.x && a.y==b.x && a.z==b.z
end

function negate(a::Vec3)::Vec3
  Vec3(-a.x, -a.y, -a.z)
end

function unit_vector(a::Vec3)::Vec3
  div(a, length(a))
end

#==============================================================
Rays
==============================================================#

type Ray
  origin::Vec3
  direction::Vec3
end

function pointAtParameter(ray::Ray, t::Float64)::Vec3
  add(ray.origin, mul(ray.direction, t))
end

#==============================================================
Image output
==============================================================#

function writePixelArrayToFile(pixels::Array{Vec3, 2})
  open("output.ppm", "w") do f
    width = size(pixels, 1)
    height = size(pixels, 2)
    write(f, "P3\n$width $height\n255\n")
    for y = reverse(1:height), x = 1:width
        ir = convert(Int64, round(255 * pixels[x, y].x))
        ig = convert(Int64, round(255 * pixels[x, y].y))
        ib = convert(Int64, round(255 * pixels[x, y].z))
        write(f, "$ir $ig $ib\n")
    end
  end
end

#==============================================================
Geometry
==============================================================#
type HitRecord
  t::Float64
  p::Vec3
  normal::Vec3
end

abstract Hitable

type Sphere <: Hitable
  center::Vec3
  radius::Float64
end

# Specialized Sphere hitable
# Returns result, hitRecord
function hit(hitable::Sphere, ray::Ray, t_min::Float64, t_max::Float64, hitRecord)::Tuple{Bool, Any}
  oc = subtract(ray.origin, hitable.center)         # Vec3
  a = dot(ray.direction, ray.direction)             # Float64
  b = dot(oc, ray.direction)                        # Float64
  c = dot(oc, oc) - hitable.radius * hitable.radius # Float64
  discriminant = b*b - a*c                          # Float64
  if discriminant > 0
    temp = (-b - sqrt(b*b-a*c)) / a
    if temp < t_max && temp > t_min
      t = temp
      p = pointAtParameter(ray, t)
      normal = div(subtract(p, hitable.center), hitable.radius)
      # hitRecord.material = hitable.material
      return (true, HitRecord(t, p, normal))
    end

    temp = (-b + sqrt(b*b-a*c)) / a
    if temp < t_max && temp > t_min
      t = temp
      p = pointAtParameter(ray, t)
      normal = div(subtract(p, hitable.center), hitable.radius)
      # hitRecord.material = hitable.material
      return (true, HitRecord(t, p, normal))
    end
  end
  return (false, nothing)
end

type HitableList <: Hitable
  list::Array{Hitable}
end

# Specialized hitable list hit function
# Returns result, hitRecord
function hit(hitable::HitableList, ray::Ray, t_min::Float64, t_max::Float64, hitRecord)::Tuple{Bool, Any}
    hitAnything = false
    closestSoFar = t_max
    tempHitRecord = nothing
    hitRecordResult = nothing
    for element in hitable.list
      result, tempHitRecord = hit(element, ray, t_min, closestSoFar, hitRecord)
      if result
        hitAnything = true
        closestSoFar = tempHitRecord.t
        hitRecordResult = tempHitRecord
      end
    end
    return (hitAnything, hitRecordResult)
end

#==============================================================
Camera
==============================================================#

type Camera
  lowerLeftCorner::Vec3
  horizontal::Vec3
  vertical::Vec3
  origin::Vec3

  Camera() = new(
    Vec3(-2.0, -1.0, -1.0),
    Vec3(4.0, 0.0, 0.0),
    Vec3(0.0, 2.0, 0.0),
    Vec3(0.0, 0.0, 0.0)
  )
end

function getRay(camera::Camera, u::Float64, v::Float64)::Ray
  direction = subtract(add(add(camera.lowerLeftCorner, mul(u, camera.horizontal)), mul(v, camera.vertical)), camera.origin)
  Ray(camera.origin, direction)
end

#==============================================================
Utils
==============================================================#

function randomInUnitSphere()::Vec3
  p = Vec3(typemax(Float64), typemax(Float64), typemax(Float64))
  while lengthSquared(p) >= 1
    p = subtract(mul(2.0, Vec3(rand(), rand(), rand())), Vec3(1,1,1))
  end
  p
end

#==============================================================
Rendering
==============================================================#

function colorFromRay(ray::Ray, world::Hitable)::Vec3
  hitRecord = nothing
  result, hitRecord = hit(world, ray, 0.001, typemax(Float64), hitRecord)
  if result
    target = add(add(hitRecord.p, hitRecord.normal), randomInUnitSphere())
    return mul(0.5, colorFromRay(Ray(hitRecord.p, subtract(target, hitRecord.p)), world))
  else
    unit_direction = unit_vector(ray.direction)
    t = 0.5 * (unit_direction.y + 1)
    return add(mul((1.0 - t), Vec3(1,1,1)), mul(t, Vec3(0.5, 0.7, 1.0)))
  end
end

function main()

  # Seed random generator
  srand(0)

  width = 200
  height = 100
  samples = 100
  pixelArray = fill(Vec3Zero(), width, height)
  lowerLeftCorner = Vec3(-2, -1, -1)
  horizontal = Vec3(4,0,0)
  vertical = Vec3(0,2,0)
  origin = Vec3(0,0,0)
  world = HitableList([
    Sphere(Vec3(0,0,-1), 0.5),
    Sphere(Vec3(0,-100.5, -1), 100)
  ])
  camera = Camera()

  for j = reverse(1:height), i = 1:width
    color = Vec3Zero()
    for sample = 1:samples
      u = Float64(i + rand()) / Float64(width)
      v = Float64(j + rand()) / Float64(height)
      ray = getRay(camera, u, v)
      color = add(color, colorFromRay(ray, world))
    end
    color = div(color, Float64(samples))
    color = vec_sqrt(color) # Gamma correction
    pixelArray[i, j] = color
  end
  writePixelArrayToFile(pixelArray)
end
@time main()
exit(0)






























#==============================================================
Materials
==============================================================#
abstract Material

type Lambertian <: Material
  albedo::Vec3
end

type Metal <: Material
  albedo::Vec3
  fuzz::Float64

  Metal(albedo::Vec3, fuzz::Float64) = new(albedo, fuzz < 1 ? fuzz : 1)
end

type Dialetric <: Material
  reflectiveIndex::Float64
end

function scatter(material::Lambertian, ray::Ray, hitRecord::HitRecord)::Tuple{Bool, Vec3, Ray}
  target = add(add(hitRecord.p, hitRecord.normal), randomInUnitSphere())
  attenuation = material.albedo
  scattered = Ray(hitRecord.p, subtract(target, hitRecord.p))
  return (true, attenuation, scattered)
end

function scatter(material::Metal, ray::Ray, hitRecord::HitRecord)::Tuple{Bool, Vec3, Ray}
  reflected = reflect(unit_vector(direction(ray)), hitRecord.normal)
  scattered = Ray(hitRecord.p, add(reflected, mul(material.fuzz, randomInUnitSphere())))
  attenuation = material.albedo
  result = dot(direction(scattered), hitRecord.normal) > 0
  return (result, attenuation, scattered)
end

function scatter(material::Dialetric, ray::Ray, hitRecord::HitRecord)::Tuple{Bool, Vec3, Ray}
  outwardNormal = Vec3Zero()
  ni_over_nt = Float64(0)
  reflected = reflect(ray.direction, hitRecord.normal)
  attenuation = Vec3(1,1,1)
  refracted = Vec3Zero()
  reflectProb = Float64(0)
  cosine = Float64(0)
  scattered = Ray(Vec3Zero(), Vec3Zero())

  if direction(dot(ray, hitRecord.normal)) > 0
    outwardNormal = negate(hitRecord.normal)
    ni_over_nt = material.reflectiveIndex
    cosine = div(mul(material.reflectiveIndex, dot(direction(ray), hitRecord.normal)), length(direction(ray)))
  else
    outwardNormal = hitRecord.normal
    ni_over_nt = 1.0 / reflectiveIndex
    cosine = div(negate(dot(direction(ray), hitRecord.normal)), length(direction(ray)))
  end

  result, refracted = refract(direction(ray), outwardNormal, ni_over_nt)
  if result
    reflectProb = schlick(cosine, reflectiveIndex)
  else
    reflectProb = 1
  end

  if rand() < reflectProb
    scattered = Ray(hitRecord.p, reflected)
  else
    scattered = Ray(hitRecord.p, refracted)
  end
  return (true, attenuation, scattered)
end

#==============================================================
Main raytracing code
==============================================================#



function randomInUnitDisk()::Vec3
  p = Vec3(FLOAT_MAX, FLOAT_MAX, FLOAT_MAX)
  while dot(p, p) >= 1
    p = subtract(mul(2, Vec3(rand(), rand(), 0)), Vec3(1,1,0))
  end
  p
end

function reflect(v::Vec3, n::Vec3)::Vec3
  subtract(v, mul(mul(2, dot(v, n)),n))
end

# This function either returns true and a refracted vector
# or false and Vec3Zero
function refract(v::Vec3, n::Vec3, ni_over_nt::Float64)::Tuple{Bool, Vec3}
  uv = unit_vector(v)
  dt = dot(uv, n)
  discriminant = subtract(1.0, mul(ni_over_nt ^ 2, subtract(1, mul(dt, dt))))
  if discriminant > 0
    refracted = sub(mul(ni_over_nt, subtract(uv, mul(n, dt))), mul(n, sqrt(discriminant)))
    return (true, refracted)
  end
  return (false, Vec3Zero())
end

function schlick(cosine::Float64, reflectiveIndex::Float64)::Float64
  r = (1-reflectiveIndex) * (1+reflectiveIndex)
  r = r*r
  r+(1-r)*(1-cosine)^5
end

function main()
  width = 400
  height = 200
  samples = 100
  FLOAT_MAX = typemax(Float64)
  pixelArray = fill(Vec3Zero(), width, height)

  # Seed random generator
  srand(0)

  #TODO: Implement me...
end





#==============================================================
DEBUG & TEST CODE
==============================================================#
function calc1()
  width = 10
  height = 5
  pixelArray = fill(Vec3(0,0,0), width, height)
  pixelArray[10,1] = Vec3(1,1,1)
  imageFromPixelArray(pixelArray) |> print
end
calc1()

a = Metal(Vec3(0,0,0), 1.0)
b = HitRecord(0.0, Vec3(0,0,0), Vec3(0,0,0 ), nothing)
print(b)
